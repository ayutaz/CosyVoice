# -*- coding: utf-8 -*-
"""S3 spike: peft LoRA (r=64, alpha=128) on the Qwen2 backbone of CosyVoice3.

Verifies:
  1. [trainable-count]     trainable params == 35,192,832 (r=64 over q/k/v/o/gate/up/down x 24 layers)
  2. [module-count]        168 LoRA-injected modules (24 layers x 7); lm_head / embed_tokens untouched
  3. [zero-init-identity]  direct-path (model.model) output unchanged right after wrapping (lora_B == 0)
  4. [direct-path-active]  after writing non-zero lora_B, direct-path output changes
                           (i.e. LoRA is live on the Qwen2Encoder-style `self.model.model(...)` call path)
  5. [disable-adapter]     peft_model.disable_adapter() also neutralizes the direct-path call
  6. [state-dict-roundtrip] save_pretrained -> PeftModel.from_pretrained on a fresh backbone
                           reproduces the check-4 direct-path output; record adapter file size
  7. [frozen-backbone]     every non-LoRA parameter has requires_grad == False

Run (from repo root, CPU only, ~2-4 min because the 0.5B backbone is loaded twice):
  .venv\\Scripts\\python.exe scripts\\spikes\\s3_lora_peft.py
"""
import os
import sys

import torch
from transformers import Qwen2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.tuners.lora import Linear as LoraLinear

MODEL_PATH = r"C:\Users\yuta\Desktop\Private\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B\CosyVoice-BlankEN"
SAVE_DIR = r"C:\Users\yuta\AppData\Local\Temp\s3_lora_spike_adapter"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXPECTED_TRAINABLE = 35_192_832
EXPECTED_LORA_MODULES = 24 * 7  # 168

results = {}


def check(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {detail}")


def direct_forward(causal_lm, inputs_embeds, attention_mask):
    """Mimics Qwen2Encoder.forward (cosyvoice/llm/llm.py:238): calls the Qwen2Model
    backbone directly, bypassing any wrapper's forward()."""
    with torch.no_grad():
        out = causal_lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
    return out.last_hidden_state


def main():
    torch.manual_seed(1234)

    print(f"loading backbone (float32, cpu): {MODEL_PATH}")
    base_ref = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    base_ref.eval()
    assert next(base_ref.parameters()).dtype == torch.float32

    hidden = base_ref.config.hidden_size  # 896
    n_layers = base_ref.config.num_hidden_layers  # 24

    # Fixed input, same call shape as Qwen2Encoder: (B, T, hidden) embeds + 2D bool mask
    E = torch.randn(1, 8, hidden, dtype=torch.float32)
    mask2d = torch.ones(1, 8, dtype=torch.bool)

    out_before_wrap = direct_forward(base_ref, E, mask2d)

    # ---- wrap with peft -----------------------------------------------------
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type=None,
    )
    # get_peft_model mutates base_ref's submodules in place; base_ref keeps pointing
    # at the same (now LoRA-injected) module tree -- exactly the reference that
    # Qwen2Encoder.self.model would hold in the real code.
    peft_model = get_peft_model(base_ref, lora_cfg)
    peft_model.eval()

    # ---- 1. trainable-count -------------------------------------------------
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    ok = trainable == EXPECTED_TRAINABLE
    check("trainable-count",
          ok,
          f"trainable={trainable:,} expected={EXPECTED_TRAINABLE:,} total={total:,} "
          f"(trainable {100.0 * trainable / total:.3f}%)")
    if not ok:
        print("  per-module LoRA parameter breakdown:")
        per_mod = {}
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                key = n.split(".lora_")[0].split(".")[-1]
                per_mod[key] = per_mod.get(key, 0) + p.numel()
        for k, v in sorted(per_mod.items()):
            print(f"    {k}: {v:,}")

    # ---- 2. module-count ----------------------------------------------------
    lora_modules = [n for n, m in peft_model.named_modules() if isinstance(m, LoraLayer)]
    bad = [n for n in lora_modules if ("lm_head" in n or "embed_tokens" in n)]
    per_type = {}
    for n in lora_modules:
        per_type[n.split(".")[-1]] = per_type.get(n.split(".")[-1], 0) + 1
    check("module-count",
          len(lora_modules) == EXPECTED_LORA_MODULES and not bad,
          f"lora_modules={len(lora_modules)} expected={EXPECTED_LORA_MODULES} "
          f"per_type={per_type} lm_head/embed_tokens_injected={bad or 'none'}")

    # ---- 3. zero-init-identity ----------------------------------------------
    out_after_wrap = direct_forward(base_ref, E, mask2d)
    diff3 = (out_after_wrap - out_before_wrap).abs().max().item()
    check("zero-init-identity",
          torch.equal(out_after_wrap, out_before_wrap),
          f"max_abs_diff={diff3:.3e} (exact_equal={torch.equal(out_after_wrap, out_before_wrap)})")

    # ---- 4. direct-path-active ----------------------------------------------
    torch.manual_seed(42)
    n_b = 0
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_(mean=0.0, std=0.01)
                n_b += 1
    out_perturbed = direct_forward(base_ref, E, mask2d)
    diff4 = (out_perturbed - out_before_wrap).abs().max().item()
    q0 = base_ref.model.layers[0].self_attn.q_proj
    type_ok = isinstance(q0, LoraLinear)
    check("direct-path-active",
          diff4 > 1e-4 and type_ok,
          f"max_abs_diff_vs_prewrap={diff4:.6f} lora_B_tensors_written={n_b} "
          f"layers[0].self_attn.q_proj type={type(q0).__module__}.{type(q0).__name__} "
          f"is_peft_lora_linear={type_ok}")

    # ---- 5. disable-adapter -------------------------------------------------
    with peft_model.disable_adapter():
        out_disabled = direct_forward(base_ref, E, mask2d)
    diff5 = (out_disabled - out_before_wrap).abs().max().item()
    disable_works_on_direct_path = torch.equal(out_disabled, out_before_wrap)
    check("disable-adapter",
          disable_works_on_direct_path,
          f"max_abs_diff_vs_prewrap={diff5:.3e} "
          f"(disable_adapter() DOES {'':s}affect the direct base_ref.model path: "
          f"{disable_works_on_direct_path})")
    # sanity: adapter must be live again after the context exits
    out_reenabled = direct_forward(base_ref, E, mask2d)
    assert torch.equal(out_reenabled, out_perturbed), "adapter did not re-enable after context exit"

    # ---- 6. state-dict-roundtrip ---------------------------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)
    peft_model.save_pretrained(SAVE_DIR)
    adapter_file = os.path.join(SAVE_DIR, "adapter_model.safetensors")
    size_bytes = os.path.getsize(adapter_file)
    print(f"  saved adapter: {adapter_file} ({size_bytes:,} bytes = {size_bytes / 1e6:.1f} MB)")

    print("  loading fresh backbone for roundtrip...")
    fresh_base = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    fresh_base.eval()
    fresh_peft = PeftModel.from_pretrained(fresh_base, SAVE_DIR)
    fresh_peft.eval()
    out_roundtrip = direct_forward(fresh_base, E, mask2d)
    diff6 = (out_roundtrip - out_perturbed).abs().max().item()
    check("state-dict-roundtrip",
          diff6 < 1e-6,
          f"max_abs_diff_vs_check4={diff6:.3e} exact_equal={torch.equal(out_roundtrip, out_perturbed)} "
          f"adapter_model.safetensors={size_bytes:,} bytes")

    # ---- 7. frozen-backbone --------------------------------------------------
    non_lora_trainable = [n for n, p in peft_model.named_parameters()
                          if p.requires_grad and "lora_" not in n]
    lora_trainable = sum(1 for n, p in peft_model.named_parameters()
                         if p.requires_grad and "lora_" in n)
    check("frozen-backbone",
          len(non_lora_trainable) == 0,
          f"non_lora_trainable_params={len(non_lora_trainable)} "
          f"({non_lora_trainable[:5]}) lora_trainable_tensors={lora_trainable}")

    # ---- summary --------------------------------------------------------------
    n_pass = sum(results.values())
    print(f"\nSUMMARY: {n_pass}/{len(results)} PASS")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
