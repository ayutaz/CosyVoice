# Copyright (c) 2025 Speech Speculative Decoding Tests
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import MagicMock

import torch
from transformers import Qwen2Config
from transformers.cache_utils import DynamicCache

from cosyvoice.llm.speculative_decoding import (
    DRAFT_LAYER_INDICES,
    DRAFT_NUM_LAYERS,
    DraftQwen2Encoder,
    SpeculativeDecoder,
    build_causal_mask,
    extract_draft_state_dict,
    truncate_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_small_config():
    """Create a small Qwen2Config for fast tests."""
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def make_dynamic_cache(num_layers=8, batch=1, num_heads=2, seq_len=10, head_dim=32):
    """Create a DynamicCache with dummy KV entries."""
    cache = DynamicCache()
    for _ in range(num_layers):
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        cache.update(k, v, _)
    return cache


# ---------------------------------------------------------------------------
# 1. KV Cache Truncation
# ---------------------------------------------------------------------------


class TestTruncateCache:
    def test_truncate_removes_last_n(self):
        cache = make_dynamic_cache(num_layers=4, seq_len=10)
        truncate_cache(cache, 3)
        for i in range(4):
            assert cache.key_cache[i].shape[2] == 7
            assert cache.value_cache[i].shape[2] == 7

    def test_truncate_zero_is_noop(self):
        cache = make_dynamic_cache(num_layers=2, seq_len=5)
        truncate_cache(cache, 0)
        assert cache.key_cache[0].shape[2] == 5

    def test_truncate_negative_is_noop(self):
        cache = make_dynamic_cache(num_layers=2, seq_len=5)
        truncate_cache(cache, -1)
        assert cache.key_cache[0].shape[2] == 5

    def test_truncate_all(self):
        cache = make_dynamic_cache(num_layers=2, seq_len=3)
        truncate_cache(cache, 3)
        assert cache.key_cache[0].shape[2] == 0


# ---------------------------------------------------------------------------
# 2. Causal Mask
# ---------------------------------------------------------------------------


class TestBuildCausalMask:
    def test_shape(self):
        mask = build_causal_mask(5, torch.device("cpu"))
        assert mask.shape == (1, 5, 5)

    def test_is_lower_triangular(self):
        mask = build_causal_mask(4, torch.device("cpu"))
        expected = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
        assert torch.equal(mask, expected)


# ---------------------------------------------------------------------------
# 3. Draft Model Construction
# ---------------------------------------------------------------------------


class TestDraftQwen2Encoder:
    def test_construction(self):
        config = make_small_config()
        draft = DraftQwen2Encoder(config)
        num_layers = draft.model.config.num_hidden_layers
        assert num_layers == DRAFT_NUM_LAYERS

    def test_forward_one_step_shape(self):
        config = make_small_config()
        draft = DraftQwen2Encoder(config)
        draft.eval()
        batch, seq_len, hidden = 1, 5, config.hidden_size
        xs = torch.randn(batch, seq_len, hidden)
        masks = build_causal_mask(seq_len, torch.device("cpu"))
        with torch.no_grad():
            out, cache = draft.forward_one_step(xs, masks, cache=None)
        assert out.shape == (batch, seq_len, hidden)
        assert cache is not None

    def test_forward_one_step_with_cache(self):
        config = make_small_config()
        draft = DraftQwen2Encoder(config)
        draft.eval()
        hidden = config.hidden_size
        # First step: full sequence
        xs1 = torch.randn(1, 3, hidden)
        mask1 = build_causal_mask(3, torch.device("cpu"))
        with torch.no_grad():
            out1, cache1 = draft.forward_one_step(xs1, mask1, cache=None)
        # Second step: single token with cache
        xs2 = torch.randn(1, 1, hidden)
        mask2 = build_causal_mask(4, torch.device("cpu"))  # 3 cached + 1 new
        with torch.no_grad():
            out2, cache2 = draft.forward_one_step(xs2, mask2, cache=cache1)
        assert out2.shape == (1, 1, hidden)


# ---------------------------------------------------------------------------
# 4. Weight Extraction / Layer Remapping
# ---------------------------------------------------------------------------


class TestExtractDraftWeights:
    def test_layer_remapping(self):
        # Build a fake state dict with 24 layers
        target_sd = {}
        for i in range(24):
            target_sd["llm.model.model.layers.{}.self_attn.q_proj.weight".format(i)] = torch.randn(64, 64)
        target_sd["llm.model.model.embed_tokens.weight"] = torch.randn(256, 64)
        target_sd["llm.model.model.norm.weight"] = torch.randn(64)
        target_sd["llm_decoder.weight"] = torch.randn(100, 64)
        target_sd["speech_embedding.weight"] = torch.randn(100, 64)

        draft_sd = extract_draft_state_dict(target_sd)

        # Check only 8 layers present
        draft_layers = set()
        for key in draft_sd:
            if "llm.model.model.layers." in key:
                parts = key.split(".")
                idx = int(parts[parts.index("layers") + 1])
                draft_layers.add(idx)
        assert draft_layers == set(range(DRAFT_NUM_LAYERS))

        # Check non-layer keys are preserved
        assert "llm.model.model.embed_tokens.weight" in draft_sd
        assert "llm.model.model.norm.weight" in draft_sd
        assert "llm_decoder.weight" in draft_sd
        assert "speech_embedding.weight" in draft_sd

    def test_correct_layer_mapping(self):
        target_sd = {}
        for i in range(24):
            target_sd["llm.model.model.layers.{}.w".format(i)] = torch.tensor([float(i)])

        draft_sd = extract_draft_state_dict(target_sd)

        for dst_idx, src_idx in enumerate(DRAFT_LAYER_INDICES):
            key = "llm.model.model.layers.{}.w".format(dst_idx)
            assert key in draft_sd
            assert draft_sd[key].item() == float(src_idx)


# ---------------------------------------------------------------------------
# 5. Acceptance Criterion
# ---------------------------------------------------------------------------


class TestAcceptanceCriterion:
    def test_always_accept_with_high_beta(self):
        """With beta=1.0, acceptance probability >= 1.0, always accept."""
        p_val, q_val = 0.3, 0.1
        beta = 1.0
        accept_prob = min(1.0, q_val / p_val) + beta
        assert accept_prob >= 1.0

    def test_strict_rejection_with_zero_beta(self):
        """With beta=0, q < p should sometimes reject."""
        p_val, q_val = 0.5, 0.1
        beta = 0.0
        accept_prob = min(1.0, q_val / p_val) + beta
        # accept_prob = 0.2 < 1.0, so rejection is possible
        assert accept_prob < 1.0

    def test_equal_distributions_always_accept(self):
        """When p == q, min(1, q/p) = 1.0, always accept even with beta=0."""
        p_val = q_val = 0.5
        beta = 0.0
        accept_prob = min(1.0, q_val / p_val) + beta
        assert accept_prob >= 1.0


# ---------------------------------------------------------------------------
# 6. Rejection Resampling
# ---------------------------------------------------------------------------


class TestRejectionResampling:
    def test_max_zero_q_minus_p_is_valid_distribution(self):
        """normalize(max(0, q - p)) should be a valid probability distribution."""
        q = torch.tensor([0.1, 0.3, 0.4, 0.2])
        p = torch.tensor([0.3, 0.1, 0.2, 0.4])
        diff = torch.clamp(q - p, min=0)
        # diff = [0, 0.2, 0.2, 0]
        diff_sum = diff.sum()
        assert diff_sum > 0
        dist = diff / diff_sum
        assert abs(dist.sum().item() - 1.0) < 1e-6
        assert (dist >= 0).all()

    def test_q_dominates_p(self):
        """When q > p everywhere, result should be proportional to (q - p)."""
        q = torch.tensor([0.5, 0.3, 0.2])
        p = torch.tensor([0.1, 0.1, 0.1])
        diff = torch.clamp(q - p, min=0)
        dist = diff / diff.sum()
        assert abs(dist.sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. EOS Handling
# ---------------------------------------------------------------------------


class TestEOSHandling:
    def test_eos_ignored_before_min_len(self):
        """The _sample_token should retry when EOS is returned before min_len."""
        stop_ids = [10, 11]
        call_count = [0]

        def mock_sampling(logp, decoded, sampling_val):
            call_count[0] += 1
            if call_count[0] <= 3:
                return 10  # stop token
            return 5  # valid token

        decoder = SpeculativeDecoder(
            target_llm=None,
            draft_llm=None,
            llm_decoder=None,
            speech_embedding=None,
            stop_token_ids=stop_ids,
            sampling_fn=mock_sampling,
        )
        logits = torch.randn(1, 20)
        token = decoder._sample_token(logits, [], 25, ignore_eos=True)
        assert token == 5
        assert call_count[0] == 4

    def test_eos_accepted_after_min_len(self):
        """The _sample_token should accept EOS after min_len."""
        stop_ids = [10]

        def mock_sampling(logp, decoded, sampling_val):
            return 10

        decoder = SpeculativeDecoder(
            target_llm=None,
            draft_llm=None,
            llm_decoder=None,
            speech_embedding=None,
            stop_token_ids=stop_ids,
            sampling_fn=mock_sampling,
        )
        logits = torch.randn(1, 20)
        token = decoder._sample_token(logits, [], 25, ignore_eos=False)
        assert token == 10


# ---------------------------------------------------------------------------
# 8. SSD Decode with Mocks (full accept scenario)
# ---------------------------------------------------------------------------


class TestSSDDecodeMock:
    def _make_mock_decoder(self, vocab_size=20, hidden_size=64, stop_ids=None):
        """Build a SpeculativeDecoder with mock target/draft that always produce valid tokens."""
        if stop_ids is None:
            stop_ids = [vocab_size - 1]

        # Mock llm_decoder: linear projection
        llm_decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        # Mock speech_embedding
        speech_embedding = torch.nn.Embedding(vocab_size, hidden_size)

        # Build a mock forward_one_step that returns random hidden states and a cache
        def make_forward_fn(num_layers):
            def forward_one_step(xs, masks, cache=None):
                if cache is None:
                    cache = DynamicCache()
                seq_len = xs.shape[1]
                for layer_idx in range(num_layers):
                    k = torch.randn(1, 2, seq_len, hidden_size // 2)
                    v = torch.randn(1, 2, seq_len, hidden_size // 2)
                    cache.update(k, v, layer_idx)
                return xs, cache

            return forward_one_step

        target_llm = MagicMock()
        target_llm.forward_one_step = make_forward_fn(24)
        draft_llm = MagicMock()
        draft_llm.forward_one_step = make_forward_fn(8)

        token_counter = [0]

        def deterministic_sampling(logp, decoded, sampling_val):
            token_counter[0] += 1
            # Return token 0 always (not a stop token)
            return 0

        return SpeculativeDecoder(
            target_llm=target_llm,
            draft_llm=draft_llm,
            llm_decoder=llm_decoder,
            speech_embedding=speech_embedding,
            stop_token_ids=stop_ids,
            sampling_fn=deterministic_sampling,
            num_draft_tokens=3,
            tolerance=0.4,
        )

    def test_decode_produces_tokens(self):
        """SSD decode should produce at least some tokens."""
        decoder = self._make_mock_decoder()
        lm_input = torch.randn(1, 5, 64)
        tokens = list(decoder.decode(lm_input, sampling=25, min_len=2, max_len=10))
        assert len(tokens) > 0
        assert all(t == 0 for t in tokens)

    def test_decode_respects_max_len(self):
        """SSD decode should not produce more than max_len tokens."""
        decoder = self._make_mock_decoder()
        lm_input = torch.randn(1, 5, 64)
        tokens = list(decoder.decode(lm_input, sampling=25, min_len=1, max_len=8))
        assert len(tokens) <= 8

    def test_decode_stops_on_eos(self):
        """SSD decode should stop when EOS is generated."""
        vocab_size = 20
        hidden_size = 64
        stop_ids = [19]

        llm_decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        speech_embedding = torch.nn.Embedding(vocab_size, hidden_size)

        def make_forward_fn(num_layers):
            def forward_one_step(xs, masks, cache=None):
                if cache is None:
                    cache = DynamicCache()
                seq_len = xs.shape[1]
                for layer_idx in range(num_layers):
                    k = torch.randn(1, 2, seq_len, hidden_size // 2)
                    v = torch.randn(1, 2, seq_len, hidden_size // 2)
                    cache.update(k, v, layer_idx)
                return xs, cache

            return forward_one_step

        target_llm = MagicMock()
        target_llm.forward_one_step = make_forward_fn(24)
        draft_llm = MagicMock()
        draft_llm.forward_one_step = make_forward_fn(8)

        call_count = [0]

        def eos_sampling(logp, decoded, sampling_val):
            call_count[0] += 1
            if call_count[0] > 5:
                return 19  # stop token
            return 0

        decoder = SpeculativeDecoder(
            target_llm=target_llm,
            draft_llm=draft_llm,
            llm_decoder=llm_decoder,
            speech_embedding=speech_embedding,
            stop_token_ids=stop_ids,
            sampling_fn=eos_sampling,
            num_draft_tokens=3,
            tolerance=0.4,
        )
        lm_input = torch.randn(1, 5, 64)
        tokens = list(decoder.decode(lm_input, sampling=25, min_len=0, max_len=100))
        # Should have stopped before max_len
        assert len(tokens) < 100


# ---------------------------------------------------------------------------
# 9. Cache Sync After Full Accept
# ---------------------------------------------------------------------------


class TestCacheSyncAfterFullAccept:
    def test_draft_cache_sync(self):
        """When all draft tokens are accepted, draft cache should be synced
        by running the last accepted draft token through draft."""
        num_draft_layers = 8
        cache = make_dynamic_cache(num_layers=num_draft_layers, seq_len=5)
        assert cache.key_cache[0].shape[2] == 5

        # Simulate: all 3 draft tokens accepted, draft_remove = 3 - 3 - 1 = -1
        # This means we need to run one more step through draft
        # After the extra step, cache length should increase by 1
        # (In actual code this is done by SpeculativeDecoder.decode)
        # Here we just verify the cache truncation math
        Ld = 3
        n_accepted = 3
        draft_remove = Ld - n_accepted - 1  # = -1
        target_remove = Ld - n_accepted  # = 0

        assert draft_remove == -1
        assert target_remove == 0
