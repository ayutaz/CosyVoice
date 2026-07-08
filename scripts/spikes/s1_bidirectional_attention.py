# -*- coding: utf-8 -*-
"""S1 spike: verify that transformers 4.51.3 Qwen2 supports bidirectional attention
via a user-supplied 4D additive attention mask.

Background (DELTA-TTS reproduction): CosyVoice3's Qwen2 backbone is called as
    model.model(inputs_embeds=xs, attention_mask=masks, use_cache=False)
(cosyvoice/llm/llm.py:238). We verify on exactly this call path that:

  1. [causal-baseline]     2D all-ones mask -> causal (perturbing last position
                           does NOT change hidden states of earlier positions).
  2. [bidirectional-4d]    all-visible additive 4D float mask (zeros) -> bidirectional
                           (perturbing last position DOES change earlier positions).
  3. [eager-sdpa-parity]   eager vs sdpa give the same output with the same 4D mask.
  4. [padding-composition] masking key-side columns (additive min_dtype) in the 4D
                           mask isolates valid positions from padding positions.
  5. [code-path]           (documented below / in the spike report) the 4D mask is
                           passed through unchanged:
       - Qwen2Model.forward -> _update_causal_mask (modeling_qwen2.py:519, :579)
       - sdpa: AttentionMaskConverter._ignore_causal_mask_sdpa returns False for
         dim==4 masks (modeling_attn_mask_utils.py:286-287) -> mask NOT dropped
       - _prepare_4d_causal_attention_mask_with_cache_position: "if attention_mask
         is not None and attention_mask.dim() == 4: causal_mask = attention_mask"
         (modeling_qwen2.py:698-700) -> used as-is
       - eager_attention_forward adds it to logits (modeling_qwen2.py:123-125)
       - sdpa_attention_forward passes it as attn_mask; is_causal fallback only
         happens when the mask is None (integrations/sdpa_attention.py:46-47)

Run (CPU, fp32):
  .venv/Scripts/python.exe scripts/spikes/s1_bidirectional_attention.py
"""
import os
import sys
import time

import torch
import transformers
from transformers import Qwen2ForCausalLM

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "pretrained_models", "Fun-CosyVoice3-0.5B", "CosyVoice-BlankEN",
)
MODEL_DIR = os.path.normpath(MODEL_DIR)

B, T, H = 1, 16, 896
N_PAD = 4                 # last 4 positions treated as padding in check 4
VALID = T - N_PAD         # 12
DTYPE = torch.float32

RESULTS = []


def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"[{name}] {status}  {detail}", flush=True)
    RESULTS.append((name, passed, detail))


def load_model(attn_impl):
    t0 = time.time()
    model = Qwen2ForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        attn_implementation=attn_impl,
    )
    model.eval()
    print(f"loaded {attn_impl} model in {time.time() - t0:.1f}s "
          f"(_attn_implementation={model.config._attn_implementation})", flush=True)
    return model


def backbone_forward(model, inputs_embeds, attention_mask):
    """Exactly the call path used in cosyvoice/llm/llm.py:238."""
    out = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return out.last_hidden_state


def main():
    torch.manual_seed(1234)
    print(f"transformers={transformers.__version__} torch={torch.__version__}", flush=True)
    print(f"model dir: {MODEL_DIR}", flush=True)

    # Fixed inputs shared by all checks.
    E = torch.randn(B, T, H, dtype=DTYPE) * 0.02          # embedding-scale inputs
    E_pert_last = E.clone()
    E_pert_last[:, -1, :] += torch.randn(H, dtype=DTYPE) * 0.1  # perturb last position only
    E_pert_pad = E.clone()
    E_pert_pad[:, VALID:, :] += torch.randn(N_PAD, H, dtype=DTYPE) * 0.1  # perturb padding positions

    mask_2d = torch.ones(B, T, dtype=torch.long)          # standard 2D mask -> causal
    mask_4d_full = torch.zeros(B, 1, T, T, dtype=DTYPE)   # additive, all-visible -> bidirectional
    min_dtype = torch.finfo(DTYPE).min
    mask_4d_pad = mask_4d_full.clone()
    mask_4d_pad[:, :, :, VALID:] = min_dtype              # key-side padding columns masked

    with torch.no_grad():
        # ---------------- eager model: checks 1, 2, 4 + reference for 3 ----------------
        model = load_model("eager")

        # Check 1: causal baseline with 2D all-ones mask.
        h_base = backbone_forward(model, E, mask_2d)
        h_pert = backbone_forward(model, E_pert_last, mask_2d)
        diff_prefix = (h_base[:, : T - 1] - h_pert[:, : T - 1]).abs().max().item()
        diff_last = (h_base[:, -1] - h_pert[:, -1]).abs().max().item()
        ok = diff_prefix <= 1e-6 and diff_last > 1e-3
        report(
            "causal-baseline",
            ok,
            f"prefix(0..{T-2}) max|dh|={diff_prefix:.3e} (expect <=1e-6, causal), "
            f"last pos max|dh|={diff_last:.3e} (sanity: perturbation visible at pos {T-1})",
        )

        # Check 2: bidirectional via all-visible 4D additive mask.
        h_bi_base = backbone_forward(model, E, mask_4d_full)
        h_bi_pert = backbone_forward(model, E_pert_last, mask_4d_full)
        diff_prefix_bi = (h_bi_base[:, : T - 1] - h_bi_pert[:, : T - 1]).abs().max().item()
        # extra sanity: bidirectional output differs from causal output on the same input
        diff_vs_causal = (h_bi_base - h_base).abs().max().item()
        ok = diff_prefix_bi > 1e-3
        report(
            "bidirectional-4d",
            ok,
            f"prefix(0..{T-2}) max|dh|={diff_prefix_bi:.3e} (expect >1e-3, info flows backward); "
            f"bidir-vs-causal same-input max|dh|={diff_vs_causal:.3e}",
        )

        # Check 4 (eager): padding columns masked out; valid positions must be
        # unaffected by changes to padding-position inputs.
        h_pad_base = backbone_forward(model, E, mask_4d_pad)
        h_pad_pert = backbone_forward(model, E_pert_pad, mask_4d_pad)
        diff_valid = (h_pad_base[:, :VALID] - h_pad_pert[:, :VALID]).abs().max().item()
        diff_padpos = (h_pad_base[:, VALID:] - h_pad_pert[:, VALID:]).abs().max().item()
        # extra sanity: valid region is still bidirectional under the padding mask
        E_pert_v = E.clone()
        E_pert_v[:, VALID - 1, :] += torch.randn(H, dtype=DTYPE) * 0.1
        h_pad_pv = backbone_forward(model, E_pert_v, mask_4d_pad)
        diff_valid_bi = (h_pad_base[:, : VALID - 1] - h_pad_pv[:, : VALID - 1]).abs().max().item()
        ok = diff_valid <= 1e-5 and diff_padpos > 1e-3 and diff_valid_bi > 1e-3
        report(
            "padding-composition",
            ok,
            f"valid(0..{VALID-1}) max|dh|={diff_valid:.3e} (expect <=1e-5, isolated from padding); "
            f"pad({VALID}..{T-1}) max|dh|={diff_padpos:.3e} (sanity: pad positions did change); "
            f"valid-region backward-flow max|dh|={diff_valid_bi:.3e} (expect >1e-3, still bidirectional)",
        )

        eager_ref_full = h_bi_base
        eager_ref_pad = h_pad_base
        del model  # free ~2GB before loading the sdpa instance

        # ---------------- sdpa model: check 3 ----------------
        model_sdpa = load_model("sdpa")
        h_sdpa_full = backbone_forward(model_sdpa, E, mask_4d_full)
        parity_full = (eager_ref_full - h_sdpa_full).abs().max().item()
        h_sdpa_pad = backbone_forward(model_sdpa, E, mask_4d_pad)
        parity_pad = (eager_ref_pad - h_sdpa_pad).abs().max().item()
        h_scale = eager_ref_full.abs().max().item()
        rel_full = parity_full / h_scale

        # Force SDPA's MATH backend (same math as eager) to separate mask-semantics
        # differences from fused-kernel accumulation differences.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                h_sdpa_math = backbone_forward(model_sdpa, E, mask_4d_full)
            parity_math = (eager_ref_full - h_sdpa_math).abs().max().item()
        except Exception as exc:  # pragma: no cover - environment dependent
            parity_math = float("nan")
            print(f"  (sdpa math-backend comparison unavailable: {exc})", flush=True)

        # Semantics are considered equivalent if either the default dispatch meets the
        # strict absolute tolerance, or the math backend does (residual = kernel accum).
        ok = parity_full < 1e-4 or parity_math < 1e-4 or rel_full < 1e-5
        report(
            "eager-sdpa-parity",
            ok,
            f"default dispatch: all-visible 4D mask max|eager-sdpa|={parity_full:.3e} "
            f"(strict target <1e-4), padding 4D mask={parity_pad:.3e}; "
            f"hidden magnitude max|h|={h_scale:.3e} -> relative={rel_full:.3e}; "
            f"SDPBackend.MATH forced: max|eager-sdpa|={parity_math:.3e}",
        )

        # sdpa fallback sanity: with a 2D all-ones mask, sdpa drops the mask
        # (_ignore_causal_mask_sdpa) and relies on is_causal=True -> must stay causal.
        h_sdpa_2d = backbone_forward(model_sdpa, E, mask_2d)
        h_sdpa_2d_p = backbone_forward(model_sdpa, E_pert_last, mask_2d)
        diff_sdpa_prefix = (h_sdpa_2d[:, : T - 1] - h_sdpa_2d_p[:, : T - 1]).abs().max().item()
        ok = diff_sdpa_prefix <= 1e-6
        report(
            "sdpa-2d-still-causal",
            ok,
            f"sdpa + 2D ones mask prefix max|dh|={diff_sdpa_prefix:.3e} "
            f"(expect <=1e-6: sdpa is_causal fallback applies only when mask is None/2D-all-ones, "
            f"never to a 4D mask)",
        )

    print("-" * 72, flush=True)
    n_fail = sum(1 for _, p, _ in RESULTS if not p)
    for name, passed, _ in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}", flush=True)
    print(f"OVERALL: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} failed)'}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
