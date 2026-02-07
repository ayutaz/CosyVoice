# Copyright (c) 2025 Speech Speculative Decoding for CosyVoice3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, Qwen2Config
from cosyvoice.utils.file_utils import logging

# Layer indices from 24-layer target to extract for 8-layer draft (paper: lower 2 + upper 6)
DRAFT_LAYER_INDICES = [0, 1, 18, 19, 20, 21, 22, 23]
DRAFT_NUM_LAYERS = len(DRAFT_LAYER_INDICES)


class DraftQwen2Encoder(torch.nn.Module):
    """8-layer draft model with same interface as Qwen2Encoder.forward_one_step."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        draft_config = Qwen2Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=DRAFT_NUM_LAYERS,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=0.0,
        )
        self.model = Qwen2ForCausalLM(draft_config)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


def truncate_cache(cache, n_remove):
    """Remove the last n_remove entries from each layer's KV cache."""
    if n_remove <= 0:
        return
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :-n_remove, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :-n_remove, :]


def build_causal_mask(seq_len, device):
    """Build a lower-triangular causal mask of shape (1, seq_len, seq_len)."""
    return torch.tril(torch.ones((1, seq_len, seq_len), device=device, dtype=torch.bool))


def extract_draft_state_dict(target_state_dict):
    """Extract and remap target model weights to draft model weights.

    Maps target layers [0, 1, 18, 19, 20, 21, 22, 23] -> draft layers [0..7].
    Copies embed_tokens, norm, and other shared weights as-is.
    """
    draft_sd = {}
    layer_map = {src: dst for dst, src in enumerate(DRAFT_LAYER_INDICES)}

    for key, value in target_state_dict.items():
        if 'llm.model.model.layers.' in key:
            parts = key.split('.')
            layer_idx_pos = parts.index('layers') + 1
            src_layer = int(parts[layer_idx_pos])
            if src_layer in layer_map:
                parts[layer_idx_pos] = str(layer_map[src_layer])
                new_key = '.'.join(parts)
                draft_sd[new_key] = value
        elif 'llm.model.' in key:
            # embed_tokens, norm, lm_head etc.
            draft_sd[key] = value
        elif key.startswith('llm_decoder.') or key.startswith('speech_embedding.'):
            draft_sd[key] = value
    return draft_sd


class SpeculativeDecoder:
    """Speech Speculative Decoding engine (Algorithm 1 from the paper).

    Args:
        target_llm: The target Qwen2Encoder (24 layers).
        draft_llm: The DraftQwen2Encoder (8 layers).
        llm_decoder: The linear projection head (shared).
        speech_embedding: The speech token embedding table (shared).
        stop_token_ids: List of token IDs that signal end of sequence.
        sampling_fn: Sampling function (e.g., ras_sampling).
        num_draft_tokens: Number of draft tokens per iteration (Ld).
        tolerance: Acceptance tolerance factor beta.
    """

    def __init__(
        self,
        target_llm,
        draft_llm,
        llm_decoder,
        speech_embedding,
        stop_token_ids,
        sampling_fn,
        num_draft_tokens=3,
        tolerance=0.4,
    ):
        self.target = target_llm
        self.draft = draft_llm
        self.llm_decoder = llm_decoder
        self.speech_embedding = speech_embedding
        self.stop_token_ids = stop_token_ids
        self.sampling_fn = sampling_fn
        self.num_draft_tokens = num_draft_tokens
        self.tolerance = tolerance

    def _sample_token(self, logits, out_tokens, sampling, ignore_eos=True):
        """Sample a single token from logits using the sampling function."""
        logp = logits.log_softmax(dim=-1)
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling_fn(logp.squeeze(dim=0), out_tokens, sampling)
            if (not ignore_eos) or (top_ids not in self.stop_token_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reached max_trials with eos when ignore_eos=True')
        return top_ids

    @torch.inference_mode()
    def decode(self, lm_input, sampling, min_len, max_len):
        """Run SSD decode loop, yielding accepted tokens one at a time.

        Args:
            lm_input: (1, prefix_len, hidden_dim) - the prefix embeddings.
            sampling: top-k value for sampling.
            min_len: minimum number of tokens before allowing EOS.
            max_len: maximum number of tokens to generate.

        Yields:
            int: accepted speech token IDs.
        """
        device = lm_input.device
        Ld = self.num_draft_tokens
        beta = self.tolerance
        out_tokens = []
        total_generated = 0

        # Phase 1: Process prefix through both target and draft, generate first token from target
        prefix_len = lm_input.shape[1]
        prefix_mask = build_causal_mask(prefix_len, device)

        target_out, target_cache = self.target.forward_one_step(lm_input, masks=prefix_mask, cache=None)
        draft_out, draft_cache = self.draft.forward_one_step(lm_input, masks=prefix_mask, cache=None)

        # Sample first token from target
        logits_first = self.llm_decoder(target_out[:, -1])
        first_token = self._sample_token(logits_first, out_tokens, sampling, ignore_eos=(total_generated < min_len))
        if first_token in self.stop_token_ids:
            return
        yield first_token
        out_tokens.append(first_token)
        total_generated += 1

        # Current input for the next step (the embedding of the last accepted token)
        current_emb = self.speech_embedding.weight[first_token].reshape(1, 1, -1)

        # Phase 2: SSD loop
        while total_generated < max_len:
            # --- A. Draft phase: generate Ld draft tokens autoregressively ---
            draft_tokens = []
            draft_probs = []
            draft_input = current_emb

            for d in range(Ld):
                draft_seq_len = draft_input.shape[1] + (draft_cache[0][0].size(2) if draft_cache is not None else 0)
                draft_mask = build_causal_mask(draft_seq_len, device)
                y_d, draft_cache = self.draft.forward_one_step(draft_input, masks=draft_mask, cache=draft_cache)
                p_d = self.llm_decoder(y_d[:, -1]).softmax(dim=-1)
                draft_probs.append(p_d)
                # Simple top-k sampling for draft (no EOS filtering, we handle EOS in verification)
                logp_d = p_d.log()
                x_d = self.sampling_fn(logp_d.squeeze(dim=0), out_tokens + draft_tokens, sampling)
                draft_tokens.append(x_d)
                draft_input = self.speech_embedding.weight[x_d].reshape(1, 1, -1)

            # --- B. Verification phase: single parallel forward through target ---
            # Build verification input: [current_emb, emb(x_0), emb(x_1), ..., emb(x_{Ld-1})]
            verify_embs = [current_emb]
            for x_d in draft_tokens:
                verify_embs.append(self.speech_embedding.weight[x_d].reshape(1, 1, -1))
            verify_input = torch.cat(verify_embs, dim=1)  # (1, Ld+1, hidden)

            verify_seq_len = verify_input.shape[1] + (target_cache[0][0].size(2) if target_cache is not None else 0)
            verify_mask = build_causal_mask(verify_seq_len, device)
            y_t, target_cache = self.target.forward_one_step(verify_input, masks=verify_mask, cache=target_cache)

            # Extract target probabilities for each position
            # y_t[:, 0] corresponds to current_emb position -> verifies draft_tokens[0]
            # y_t[:, i] verifies draft_tokens[i] (or bonus for i == Ld)
            target_probs = []
            for j in range(Ld + 1):
                q_j = self.llm_decoder(y_t[:, j]).softmax(dim=-1)
                target_probs.append(q_j)

            # --- C. Accept/reject with tolerance beta ---
            n_accepted = 0
            for j in range(Ld):
                x_j = draft_tokens[j]
                p_j = draft_probs[j]  # (1, vocab)
                q_j = target_probs[j]  # (1, vocab)

                r = torch.rand(1, device=device).item()
                # Acceptance criterion: r < min(1, q/p) + beta
                p_val = p_j[0, x_j].item()
                q_val = q_j[0, x_j].item()
                if p_val > 0:
                    accept_prob = min(1.0, q_val / p_val) + beta
                else:
                    accept_prob = 1.0 + beta  # always accept if draft assigned 0 prob

                if r < accept_prob:
                    # Accept this draft token
                    if x_j in self.stop_token_ids and total_generated >= min_len:
                        # Accepted an EOS token - stop generation
                        # Fix caches before returning
                        target_remove = Ld - n_accepted
                        if target_remove > 0:
                            truncate_cache(target_cache, target_remove)
                        return
                    elif x_j in self.stop_token_ids:
                        # Before min_len: reject EOS, resample
                        diff = torch.clamp(q_j - p_j, min=0)
                        # Zero out stop token positions
                        for sid in self.stop_token_ids:
                            diff[0, sid] = 0.0
                        diff_sum = diff.sum()
                        if diff_sum > 0:
                            diff = diff / diff_sum
                            x_prime = torch.multinomial(diff.squeeze(0), 1).item()
                        else:
                            # Fallback: sample from target, excluding stop tokens
                            q_clean = q_j.clone()
                            for sid in self.stop_token_ids:
                                q_clean[0, sid] = 0.0
                            q_clean = q_clean / q_clean.sum()
                            x_prime = torch.multinomial(q_clean.squeeze(0), 1).item()
                        yield x_prime
                        out_tokens.append(x_prime)
                        total_generated += 1
                        n_accepted = j + 1  # count as "processed" for cache trimming
                        break
                    else:
                        yield x_j
                        out_tokens.append(x_j)
                        total_generated += 1
                        n_accepted += 1
                else:
                    # Reject: sample from normalized max(0, q - p)
                    diff = torch.clamp(q_j - p_j, min=0)
                    diff_sum = diff.sum()
                    if diff_sum > 0:
                        diff = diff / diff_sum
                        x_prime = torch.multinomial(diff.squeeze(0), 1).item()
                    else:
                        # Fallback: sample from target distribution
                        x_prime = torch.multinomial(q_j.squeeze(0), 1).item()

                    if x_prime in self.stop_token_ids and total_generated >= min_len:
                        target_remove = Ld - n_accepted
                        if target_remove > 0:
                            truncate_cache(target_cache, target_remove)
                        return
                    elif x_prime in self.stop_token_ids:
                        # Before min_len, try to avoid stop tokens
                        diff_clean = torch.clamp(q_j - p_j, min=0)
                        for sid in self.stop_token_ids:
                            diff_clean[0, sid] = 0.0
                        if diff_clean.sum() > 0:
                            diff_clean = diff_clean / diff_clean.sum()
                            x_prime = torch.multinomial(diff_clean.squeeze(0), 1).item()
                        else:
                            q_clean = q_j.clone()
                            for sid in self.stop_token_ids:
                                q_clean[0, sid] = 0.0
                            q_clean = q_clean / q_clean.sum()
                            x_prime = torch.multinomial(q_clean.squeeze(0), 1).item()

                    yield x_prime
                    out_tokens.append(x_prime)
                    total_generated += 1
                    n_accepted = j  # j-th token was rejected, but we emitted a replacement
                    # For cache trimming, the replacement sits at position j
                    n_accepted = j + 1
                    break
            else:
                # All Ld draft tokens accepted -> bonus token from target
                bonus_logits = target_probs[Ld]  # (1, vocab)
                bonus_logp = bonus_logits.log()
                ignore_eos = total_generated < min_len
                bonus_token = self._sample_token(self.llm_decoder(y_t[:, -1]), out_tokens, sampling, ignore_eos=ignore_eos)
                if bonus_token in self.stop_token_ids:
                    return
                yield bonus_token
                out_tokens.append(bonus_token)
                total_generated += 1
                n_accepted = Ld

            # --- D. KV cache correction ---
            target_remove = Ld - n_accepted
            draft_remove = Ld - n_accepted - 1

            if target_remove > 0:
                truncate_cache(target_cache, target_remove)

            if draft_remove > 0:
                truncate_cache(draft_cache, draft_remove)
            elif draft_remove < 0:
                # All draft tokens accepted + bonus: draft cache is behind by 1
                # Need to feed the last accepted draft token through draft to sync cache
                last_draft_emb = self.speech_embedding.weight[draft_tokens[-1]].reshape(1, 1, -1)
                draft_seq_len = 1 + draft_cache[0][0].size(2)
                draft_mask = build_causal_mask(draft_seq_len, device)
                _, draft_cache = self.draft.forward_one_step(last_draft_emb, masks=draft_mask, cache=draft_cache)

            # Update current_emb for next iteration
            last_accepted = out_tokens[-1]
            current_emb = self.speech_embedding.weight[last_accepted].reshape(1, 1, -1)

        logging.info('SSD decode reached max_len {}'.format(max_len))
