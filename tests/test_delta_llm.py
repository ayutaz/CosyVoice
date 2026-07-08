# Tests for the DELTA-TTS masked diffusion LM (cosyvoice/llm/diffusion_llm.py).
# The 0.5B backbone is never loaded: a tiny Qwen2 (hidden 64, 2 layers) stands in.
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cosyvoice.llm.diffusion_llm import (
    FALLBACK_TOKEN_TEXT_RATIO,
    ConformerConvModule,
    DiffusionCosyVoice3LM,
    build_bidirectional_mask,
    shift_align,
    unmask_schedule,
)
from cosyvoice.utils.common import IGNORE_ID

HIDDEN = 64
VOCAB = 256
SPEECH_TOKEN_SIZE = 50


@pytest.fixture(scope='session')
def tiny_backbone_dir(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=512)
    model = Qwen2ForCausalLM(config)
    path = tmp_path_factory.mktemp('tiny_qwen2')
    model.save_pretrained(str(path))
    return str(path)


def _make_model(backbone_dir, delta=False, **kwargs):
    from cosyvoice.llm.llm import Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling
    torch.manual_seed(1)
    model = DiffusionCosyVoice3LM(
        llm_input_size=HIDDEN,
        llm_output_size=HIDDEN,
        speech_token_size=SPEECH_TOKEN_SIZE,
        llm=Qwen2Encoder(backbone_dir),
        sampling=ras_sampling,
        **kwargs,
    )
    if delta:
        model.apply_lora()
        model.attach_conv_modules()
        model.init_mask_embedding()
        model.freeze_for_delta()
    return model


class TestUnmaskSchedule:
    """(1) k_n >= 0 and sum == L for every L/T/mu, incl. the mu=0.3 floating point trap."""

    @pytest.mark.parametrize('L,T,mu', [
        (100, 16, 0.3),   # c_T = 0.9999999999999998 in fp, naive floor would drop a token
        (1, 16, 0.3),
        (0, 4, 0.3),
        (7, 3, 1.0),
        (1000, 16, 0.3),
        (37, 16, 0.3),
        (16, 16, 0.3),
        (5, 16, 0.3),     # more steps than tokens: most k_n are 0
        (50, 50, 0.5),
        (3, 8, 2.0),
        (24, 1, 0.3),     # single step commits everything
    ])
    def test_sum_and_nonnegative(self, L, T, mu):
        ks = unmask_schedule(L, T, mu)
        assert len(ks) == T
        assert all(isinstance(k, int) and k >= 0 for k in ks)
        assert sum(ks) == L

    def test_mu_03_floating_point_trap_exists(self):
        # document the trap: at n == T the schedule lands just below 1.0 in fp,
        # so floor(c_T * L) == L - 1 and the final step must be forced
        s = 1.0
        c_final = 0.3 * s / (1.0 + (0.3 - 1.0) * s)
        assert c_final < 1.0
        assert math.floor(c_final * 100) == 99
        ks = unmask_schedule(100, 16, 0.3)
        assert sum(ks) == 100

    def test_prefix_sums_telescope(self):
        # for n < T the committed count telescopes to floor(c_n * L) exactly
        L, T, mu = 137, 16, 0.3
        ks = unmask_schedule(L, T, mu)
        committed = 0
        for n in range(1, T):
            committed += ks[n - 1]
            s = n / T
            c_n = mu * s / (1.0 + (mu - 1.0) * s)
            assert committed == math.floor(c_n * L)

    @pytest.mark.parametrize('L,T,mu', [(10, 0, 0.3), (-1, 4, 0.3), (10, 4, 0.0), (10, 4, -1.0)])
    def test_invalid_args_raise(self, L, T, mu):
        with pytest.raises(ValueError):
            unmask_schedule(L, T, mu)


class TestShiftAlign:
    """(2) the target at masked position j must pair with the logits at position j-1."""

    def test_index_alignment(self):
        B, T, V = 2, 5, 4
        # logits[b, j, v] = 100*b + 10*j + v: each row identifies its (b, j) origin
        logits = (100 * torch.arange(B).reshape(B, 1, 1)
                  + 10 * torch.arange(T).reshape(1, T, 1)
                  + torch.arange(V).reshape(1, 1, V)).float()
        targets = 100 * torch.arange(B).reshape(B, 1) + torch.arange(T).reshape(1, T)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, 2] = mask[0, 4] = mask[1, 1] = True

        sl_flat, tgt_flat = shift_align(logits, targets, mask)
        assert sl_flat.shape == (3, V)
        assert tgt_flat.shape == (3,)
        # boolean indexing is row-major: (b=0, j=2), (b=0, j=4), (b=1, j=1)
        for row, (b, j) in enumerate([(0, 2), (0, 4), (1, 1)]):
            assert torch.equal(sl_flat[row], logits[b, j - 1]), 'masked position {} must read logits {}'.format(j, j - 1)
            assert tgt_flat[row] == targets[b, j]

    def test_position_zero_mask_is_dropped(self):
        # position 0 (sos) has no preceding logits, shift_align must ignore it
        logits = torch.randn(1, 4, 3)
        targets = torch.arange(4).reshape(1, 4)
        mask = torch.tensor([[True, False, True, False]])
        sl_flat, tgt_flat = shift_align(logits, targets, mask)
        assert sl_flat.shape == (1, 3)
        assert tgt_flat.tolist() == [2]

    def test_empty_mask(self):
        sl_flat, tgt_flat = shift_align(torch.randn(2, 5, 3), torch.zeros(2, 5, dtype=torch.long),
                                        torch.zeros(2, 5, dtype=torch.bool))
        assert sl_flat.shape == (0, 3)
        assert tgt_flat.shape == (0,)


class TestDeltaSequenceMasking:
    """(3) prompt region and visible positions keep their embeddings, only [M] positions get mask_emb."""

    def test_build_delta_sequence(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir)
        with torch.no_grad():
            model.mask_emb.fill_(5.0)  # distinctive, cannot collide with random embeddings
        text_token = torch.randint(0, VOCAB, (3,))
        speech_token = torch.randint(0, SPEECH_TOKEN_SIZE, (8,))
        text_emb = model.llm.model.model.embed_tokens(text_token)
        speech_emb = model.speech_embedding(speech_token)
        prompt_len = 3
        target_mask = torch.tensor([True, False, True, True, False])

        seq_emb, token_ids, seq_mask = model._build_delta_sequence(text_emb, speech_emb, speech_token,
                                                                   prompt_len, target_mask)
        Lt, Ls = 3, 8
        offset = 2 + Lt  # [sos, text x3, task_id, speech x8, eos]
        assert seq_emb.shape == (Lt + Ls + 3, HIDDEN)
        assert token_ids.shape == (Lt + Ls + 3,)
        assert seq_mask.shape == (Lt + Ls + 3,)

        # layout: sos / text / task_id / eos embeddings follow the CosyVoice3LM rows
        assert torch.equal(seq_emb[0], model.speech_embedding.weight[model.sos])
        assert torch.equal(seq_emb[1:1 + Lt], text_emb)
        assert torch.equal(seq_emb[1 + Lt], model.speech_embedding.weight[model.task_id])
        assert torch.equal(seq_emb[-1], model.speech_embedding.weight[model.eos_token])

        # speech region: prompt + visible positions unchanged, [M] positions replaced
        masked_abs = {3, 5, 6}  # prompt_len + relative index of True in target_mask
        for j in range(Ls):
            if j in masked_abs:
                assert torch.equal(seq_emb[offset + j], model.mask_emb)
                assert bool(seq_mask[offset + j])
            else:
                assert torch.equal(seq_emb[offset + j], speech_emb[j]), 'visible position {} must be unchanged'.format(j)
                assert not bool(seq_mask[offset + j])

        # mask covers speech positions only
        assert seq_mask.sum().item() == len(masked_abs)
        # token ids: IGNORE outside speech, real ids at every speech position, eos at the end
        assert (token_ids[:offset] == IGNORE_ID).all()
        assert torch.equal(token_ids[offset:offset + Ls], speech_token)
        assert token_ids[-1].item() == model.eos_token

    def test_init_mask_embedding_uses_speech_mean(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir)
        model.init_mask_embedding()
        expected = model.speech_embedding.weight[:SPEECH_TOKEN_SIZE].mean(dim=0)
        assert torch.allclose(model.mask_emb, expected)


class TestBidirectionalMask:
    """(4) padding key columns blocked, valid region fully bidirectional."""

    def test_mask_values(self):
        pad_mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
        mask = build_bidirectional_mask(pad_mask, torch.float32)
        neg = torch.finfo(torch.float32).min
        assert mask.shape == (2, 1, 5, 5)
        # sample 0 has no padding: everything visible, including future keys (bidirectional)
        assert (mask[0] == 0.0).all()
        # sample 1: padded key columns blocked for every query
        assert (mask[1, 0, :, 3:] == neg).all()
        assert (mask[1, 0, :, :3] == 0.0).all()
        # padded QUERY rows are not fully blocked (all-min rows would softmax to NaN)
        assert (mask[1, 0, 3, :3] == 0.0).all()
        assert (mask[1, 0, 4, :3] == 0.0).all()

    def test_rejects_non_bool_mask(self):
        with pytest.raises(AssertionError):
            build_bidirectional_mask(torch.ones(1, 4), torch.float32)

    def test_backbone_information_flow_is_bidirectional(self, tiny_backbone_dir):
        # a 4D additive mask must bypass the HF causal mask: editing a FUTURE input
        # position must change the output at position 0 (it does not on the causal path)
        from cosyvoice.llm.llm import Qwen2Encoder
        torch.manual_seed(0)
        encoder = Qwen2Encoder(tiny_backbone_dir)
        encoder.eval()
        x1 = torch.randn(1, 6, HIDDEN)
        x2 = x1.clone()
        x2[0, -1] += 1.0
        mask4d = build_bidirectional_mask(torch.ones(1, 6, dtype=torch.bool), x1.dtype)
        mask2d = torch.ones(1, 6, dtype=torch.bool)
        with torch.no_grad():
            bi1 = encoder.model.model(inputs_embeds=x1, attention_mask=mask4d, use_cache=False).last_hidden_state
            bi2 = encoder.model.model(inputs_embeds=x2, attention_mask=mask4d, use_cache=False).last_hidden_state
            causal1 = encoder.model.model(inputs_embeds=x1, attention_mask=mask2d, use_cache=False).last_hidden_state
            causal2 = encoder.model.model(inputs_embeds=x2, attention_mask=mask2d, use_cache=False).last_hidden_state
        assert not torch.allclose(bi1[0, 0], bi2[0, 0])
        assert torch.allclose(causal1[0, 0], causal2[0, 0])


class TestConformerConvModule:
    """(5) shape preservation, zero-init identity, no temporal leak from padding."""

    def test_shape_preserved(self):
        module = ConformerConvModule(32)
        x = torch.randn(2, 10, 32)
        assert module(x).shape == (2, 10, 32)
        assert module(x, torch.ones(2, 10, dtype=torch.bool)).shape == (2, 10, 32)

    def test_zero_init_identity(self):
        module = ConformerConvModule(32)
        x = torch.randn(2, 10, 32)
        pad_mask = torch.tensor([[True] * 10, [True] * 6 + [False] * 4])
        assert torch.equal(module(x), x)
        assert torch.equal(module(x, pad_mask), x)
        # after perturbing the zero-initialized conv it must stop being an identity
        with torch.no_grad():
            module.pointwise_conv2.weight.normal_()
        assert not torch.equal(module(x.clone()), x)

    def test_no_padding_leak(self):
        torch.manual_seed(0)
        module = ConformerConvModule(16)
        with torch.no_grad():
            module.pointwise_conv2.weight.normal_()  # make the conv path non-trivial
            module.pointwise_conv2.bias.normal_()
        module.eval()
        valid = 7
        pad_mask = torch.zeros(1, 12, dtype=torch.bool)
        pad_mask[0, :valid] = True
        x1 = torch.randn(1, 12, 16)
        x2 = x1.clone()
        x2[0, valid:] = torch.randn(12 - valid, 16) * 100.0  # garbage in the padding
        with torch.no_grad():
            out1 = module(x1, pad_mask)
            out2 = module(x2, pad_mask)
        assert torch.equal(out1[0, :valid], out2[0, :valid]), 'padding content leaked into valid positions'
        # without the pad mask the depthwise conv DOES leak, proving the mask is load-bearing
        with torch.no_grad():
            leak1 = module(x1)
            leak2 = module(x2)
        assert not torch.equal(leak1[0, :valid], leak2[0, :valid])

    def test_parameter_budget_matches_paper(self):
        # d=896: ~2.44M params per module, 24 modules ~= 58.6M vs the paper's 59M conv budget
        module = ConformerConvModule(896)
        per_layer = sum(p.numel() for p in module.parameters())
        assert per_layer == 2443392
        total = per_layer * 24
        assert total == 58641408
        assert abs(total - 59e6) / 59e6 < 0.01

    def test_even_kernel_rejected(self):
        with pytest.raises(AssertionError):
            ConformerConvModule(32, kernel_size=30)


class TestFreezeForDelta:
    """(6) after freeze_for_delta only lora / conv_modules / mask_emb are trainable."""

    def test_trainable_set(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        for name, param in model.named_parameters():
            expected = 'lora_' in name or name.startswith('conv_modules.') or name == 'mask_emb'
            assert param.requires_grad == expected, name

        summary = model.trainable_parameter_summary()
        assert summary['other'] == 0
        assert summary['lora'] > 0
        assert summary['mask_emb'] == HIDDEN
        assert summary['conv'] == sum(p.numel() for p in model.conv_modules.parameters())
        # the frozen core explicitly
        assert not model.llm_decoder.weight.requires_grad
        assert not model.speech_embedding.weight.requires_grad
        assert not model.llm.model.model.embed_tokens.weight.requires_grad

    def test_state_dict_has_no_duplicates(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        keys = list(model.state_dict().keys())
        # peft_model wraps the already-registered backbone, it must not add keys
        assert not any(k.startswith('peft_model.') for k in keys)
        # conv parameters live under conv_modules.* only, not under the wrapped layers
        conv_keys = [k for k in keys if 'pointwise_conv' in k or 'depthwise_conv' in k]
        assert len(conv_keys) > 0
        assert all(k.startswith('conv_modules.') for k in conv_keys)
        assert model.peft_model is not None

    def test_double_conversion_rejected(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        with pytest.raises(RuntimeError):
            model.apply_lora()
        with pytest.raises(RuntimeError):
            model.attach_conv_modules()


class TestDecodeSmoke:
    """(7) mini-model decode: every mask gets filled, tokens < speech_token_size, yield count == target_len."""

    def _decode(self, model, n_text, n_prompt_text, n_prompt_speech, **kwargs):
        torch.manual_seed(7)
        text = torch.randint(0, VOCAB, (1, n_text))
        prompt_text = torch.randint(0, VOCAB, (1, n_prompt_text))
        prompt_speech = torch.randint(0, SPEECH_TOKEN_SIZE, (1, n_prompt_speech))
        return list(model.inference_diffusion(
            text=text,
            text_len=torch.tensor([n_text], dtype=torch.int32),
            prompt_text=prompt_text,
            prompt_text_len=torch.tensor([n_prompt_text], dtype=torch.int32),
            prompt_speech_token=prompt_speech,
            prompt_speech_token_len=torch.tensor([n_prompt_speech], dtype=torch.int32),
            embedding=torch.zeros(0, 192),
            **kwargs,
        ))

    def test_rule_based_length_with_prompt(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        model.eval()
        tokens = self._decode(model, n_text=5, n_prompt_text=3, n_prompt_speech=9)
        assert len(tokens) == 15  # ceil(9 / 3 * 5 * 1.0)
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < SPEECH_TOKEN_SIZE for t in tokens), 'special tokens must never be sampled'

    def test_explicit_target_len(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        model.eval()
        tokens = self._decode(model, n_text=5, n_prompt_text=3, n_prompt_speech=9, target_len=7)
        assert len(tokens) == 7
        assert all(0 <= t < SPEECH_TOKEN_SIZE for t in tokens)

    def test_fallback_length_without_prompt(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        model.eval()
        tokens = self._decode(model, n_text=4, n_prompt_text=0, n_prompt_speech=0)
        assert len(tokens) == math.ceil(FALLBACK_TOKEN_TEXT_RATIO * 4)  # 24
        assert all(0 <= t < SPEECH_TOKEN_SIZE for t in tokens)

    def test_mixed_empty_prompt_falls_back(self, tiny_backbone_dir):
        # speech prompt without its transcript (cross-lingual style direct call):
        # the ratio degenerates, the documented fallback constant is used instead
        model = _make_model(tiny_backbone_dir, delta=True)
        model.eval()
        tokens = self._decode(model, n_text=4, n_prompt_text=0, n_prompt_speech=9)
        assert len(tokens) == math.ceil(FALLBACK_TOKEN_TEXT_RATIO * 4)  # not ceil(9 * 4)
        assert all(0 <= t < SPEECH_TOKEN_SIZE for t in tokens)

    def test_rule_based_length_clamped_by_max_ratio(self, tiny_backbone_dir):
        # pathological prompt (long audio, one-token transcript) must not push the
        # rule-based length past the AR-path envelope max_token_text_ratio * n_text
        model = _make_model(tiny_backbone_dir, delta=True)
        model.eval()
        tokens = self._decode(model, n_text=2, n_prompt_text=1, n_prompt_speech=40, num_steps=4)
        assert len(tokens) == 40  # min(ceil(40 / 1 * 2), ceil(20 * 2))
        # an explicit target_len is never clamped
        tokens = self._decode(model, n_text=2, n_prompt_text=1, n_prompt_speech=40, num_steps=4, target_len=50)
        assert len(tokens) == 50

    def test_num_steps_override(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        model.eval()
        tokens = self._decode(model, n_text=5, n_prompt_text=3, n_prompt_speech=9, target_len=11, num_steps=4)
        assert len(tokens) == 11
        assert all(0 <= t < SPEECH_TOKEN_SIZE for t in tokens)


class TestForwardSmoke:
    """(8) finite loss, gradients reach exactly the trainable parameters."""

    @staticmethod
    def _batch():
        torch.manual_seed(3)
        return {
            'text_token': torch.randint(0, VOCAB, (2, 6)),
            'text_token_len': torch.tensor([6, 4], dtype=torch.int32),
            'speech_token': torch.randint(0, SPEECH_TOKEN_SIZE, (2, 20)),
            'speech_token_len': torch.tensor([20, 15], dtype=torch.int32),
        }

    def test_loss_finite_and_backward_hits_trainables_only(self, tiny_backbone_dir):
        import random as _random
        model = _make_model(tiny_backbone_dir, delta=True)
        _random.seed(0)
        torch.manual_seed(0)
        out = model.forward(self._batch(), torch.device('cpu'))
        assert set(out.keys()) == {'loss', 'acc'}
        assert torch.isfinite(out['loss'])
        assert out['loss'].requires_grad
        assert 0.0 <= float(out['acc']) <= 1.0

        out['loss'].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, 'trainable {} got no grad'.format(name)
            else:
                assert param.grad is None, 'frozen {} received a grad'.format(name)
        assert model.mask_emb.grad.abs().sum() > 0
        conv_grad = sum(p.grad.abs().sum() for p in model.conv_modules.parameters())
        assert conv_grad > 0
        lora_grad = sum(p.grad.abs().sum() for n, p in model.named_parameters() if 'lora_' in n)
        assert lora_grad > 0  # lora_B grads are nonzero even while lora_B itself is still zero

    def test_degenerate_all_visible_batch(self, tiny_backbone_dir, monkeypatch):
        # every Bernoulli draw comes up visible -> zero loss with a live graph, no NaN
        import random as _random
        model = _make_model(tiny_backbone_dir, delta=True)
        _random.seed(0)
        real_rand = torch.rand
        monkeypatch.setattr(torch, 'rand', lambda *size, **kw: torch.ones(*size, device=kw.get('device')))
        try:
            out = model.forward(self._batch(), torch.device('cpu'))
        finally:
            monkeypatch.setattr(torch, 'rand', real_rand)
        assert float(out['loss'].detach()) == 0.0
        assert float(out['acc']) == 0.0
        out['loss'].backward()  # graph must stay alive

    def test_loss_weighting_normalized_by_mask_count(self, tiny_backbone_dir):
        # the loss is a 1/t weighted mean over masked positions: it must stay bounded
        # by (1/t_min) * max CE, not blow up with sequence length
        import random as _random
        model = _make_model(tiny_backbone_dir, delta=True)
        _random.seed(1)
        torch.manual_seed(1)
        out = model.forward(self._batch(), torch.device('cpu'))
        max_ce = math.log(SPEECH_TOKEN_SIZE + 200) + 20.0  # generous bound for an untrained model
        assert float(out['loss'].detach()) < (1.0 / model.t_min) * max_ce

    def test_batch_padding_isolation(self, tiny_backbone_dir):
        # integration invariant across 4D mask + conv pad mask + loss masking: with
        # deterministic masking (prompt off, t == 1 -> every target position masked,
        # 1/t weight == 1) the batch loss must equal the mask-count weighted combination
        # of the per-sample losses, i.e. a sample's loss cannot depend on the other
        # samples in the batch or on their padding
        model = _make_model(tiny_backbone_dir, delta=True,
                            prompt_ratio_max=0.0, prompt_drop=0.0, t_min=1.0)
        with torch.no_grad():
            for conv in model.conv_modules:
                conv.pointwise_conv2.weight.normal_()  # zero-init conv would hide pad-mask wiring bugs
                conv.pointwise_conv2.bias.normal_()
        model.eval()
        batch = self._batch()
        with torch.no_grad():
            loss_batch = float(model.forward(batch, torch.device('cpu'))['loss'])
            single_losses, mask_counts = [], []
            for i in range(2):
                text_len = int(batch['text_token_len'][i])
                speech_len = int(batch['speech_token_len'][i])
                single = {
                    'text_token': batch['text_token'][i:i + 1, :text_len],
                    'text_token_len': batch['text_token_len'][i:i + 1],
                    'speech_token': batch['speech_token'][i:i + 1, :speech_len],
                    'speech_token_len': batch['speech_token_len'][i:i + 1],
                }
                single_losses.append(float(model.forward(single, torch.device('cpu'))['loss']))
                mask_counts.append(speech_len)  # t == 1 masks every speech position
        combined = (single_losses[0] * mask_counts[0] + single_losses[1] * mask_counts[1]) / sum(mask_counts)
        assert loss_batch == pytest.approx(combined, abs=1e-5)


class TestARPathDisabled:
    """the inherited AR inference entry points must fail loudly, not with a bare ValueError."""

    def test_inference_and_bistream_raise_not_implemented(self, tiny_backbone_dir):
        model = _make_model(tiny_backbone_dir, delta=True)
        with pytest.raises(NotImplementedError, match='inference_diffusion'):
            model.inference()
        with pytest.raises(NotImplementedError, match='inference_diffusion'):
            model.inference_bistream()


class TestConstantWithWarmupLR:
    """train_delta.py swaps WarmupLR for a warmup-then-constant scheduler (paper: lr 1e-4 constant)."""

    @staticmethod
    def _make(warmup_steps=2000):
        from cosyvoice.bin.train_delta import ConstantWithWarmupLR
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        return opt, ConstantWithWarmupLR(opt, warmup_steps=warmup_steps)

    def test_linear_warmup(self):
        opt, sched = self._make()
        lrs = []
        for _ in range(2000):
            sched.step()
            lrs.append(opt.param_groups[0]['lr'])
        assert lrs[999] == pytest.approx(0.5e-4, rel=1e-2)  # halfway through the warmup
        assert lrs[-1] == pytest.approx(1e-4)
        assert all(b >= a for a, b in zip(lrs, lrs[1:]))

    def test_constant_after_warmup(self):
        # WarmupLR would give 4.47e-5 / 2e-5 / 1e-5 at steps 10k / 50k / 200k
        opt, sched = self._make()
        for step in (2000, 10_000, 50_000, 200_000):
            sched.set_step(step)
            sched.step()
            assert opt.param_groups[0]['lr'] == pytest.approx(1e-4), 'lr must stay at base after warmup (step {})'.format(step)
