# -*- coding: utf-8 -*-
"""S8 spike: 0.5B real-checkpoint smoke test for DiffusionCosyVoice3LM (DELTA-TTS).

Exercises the full delta conversion pipeline on the actual Fun-CosyVoice3-0.5B
checkpoint (CPU, fp32), mirroring cosyvoice/bin/train_delta.py::build_delta_model:

  1. [checkpoint-load]  build DiffusionCosyVoice3LM with the llm section args of
                        pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml, load
                        llm.pt strict=False BEFORE the delta conversion; missing keys
                        must be delta-only (lora/conv/mask_emb), unexpected must be empty.
  2. [delta-conversion] apply_lora -> attach_conv_modules -> init_mask_embedding ->
                        freeze_for_delta; trainable summary must be lora ~= 35.19M,
                        conv ~= 58.6M, mask_emb == 896, other == 0.
  3. [forward-loss]     synthetic batch (2 utterances, speech lens 60/45, English text
                        tokenized with the CosyVoice-BlankEN tokenizer) -> finite loss.
                        With zero-init conv + zero-init LoRA B the backbone is still the
                        pretrained AR model, so the masked CE lands around <= ~10.
  4. [diffusion-decode] inference_diffusion with num_steps=4, target_len=25 and a
                        20-token speech prompt -> exactly 25 yielded tokens, all < 6561.

Run (CPU, fp32, takes a few minutes):
  .venv/Scripts/python.exe scripts/spikes/s8_delta_smoke.py
"""
import math
import os
import random
import sys
import time

import torch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, 'third_party', 'Matcha-TTS')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from functools import partial  # noqa: E402

from cosyvoice.llm.llm import Qwen2Encoder  # noqa: E402
from cosyvoice.llm.diffusion_llm import DiffusionCosyVoice3LM, unmask_schedule  # noqa: E402
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  # noqa: E402
from cosyvoice.utils.common import ras_sampling  # noqa: E402

MODEL_ROOT = os.path.join(REPO_ROOT, 'pretrained_models', 'Fun-CosyVoice3-0.5B')
QWEN_DIR = os.path.join(MODEL_ROOT, 'CosyVoice-BlankEN')
LLM_PT = os.path.join(MODEL_ROOT, 'llm.pt')

SPEECH_TOKEN_SIZE = 6561
EXPECTED_LORA = 35_192_832   # r=64 on q/k/v/o/gate/up/down x 24 layers (paper: ~35.19M)
EXPECTED_CONV = 58_641_408   # 24 x ConformerConvModule(d=896, k=31) (paper: ~59M)
NUM_STEPS = 4
TARGET_LEN = 25
PROMPT_SPEECH_LEN = 20

RESULTS = []


def report(name, passed, detail):
    status = 'PASS' if passed else 'FAIL'
    print('[{}] {}  {}'.format(name, status, detail), flush=True)
    RESULTS.append((name, passed))


def is_delta_key(key):
    return key == 'mask_emb' or 'lora_' in key or key.startswith('conv_modules.')


def main():
    torch.manual_seed(1986)
    random.seed(1986)
    print('torch={} device=cpu dtype=float32'.format(torch.__version__), flush=True)
    print('model root: {}'.format(MODEL_ROOT), flush=True)

    # ---- 1. build with the cosyvoice3.yaml llm-section args, load llm.pt strict=False ----
    t0 = time.time()
    model = DiffusionCosyVoice3LM(
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=SPEECH_TOKEN_SIZE,
        llm=Qwen2Encoder(QWEN_DIR),
        sampling=partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1),
        length_normalized_loss=True,
        lsm_weight=0,
        mix_ratio=[5, 15],
    )
    state_dict = torch.load(LLM_PT, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('load llm.pt: {} tensors, missing={}, unexpected={} ({:.1f}s)'.format(len(state_dict), missing, unexpected, time.time() - t0), flush=True)
    bad_missing = [k for k in missing if not is_delta_key(k)]
    report(
        'checkpoint-load',
        len(bad_missing) == 0 and len(unexpected) == 0,
        'missing={} (all delta-only: {}), unexpected={}'.format(missing, len(bad_missing) == 0, unexpected),
    )

    # ---- 2. delta conversion + trainable summary ----
    t0 = time.time()
    model.apply_lora()
    model.attach_conv_modules()
    model.init_mask_embedding()
    model.freeze_for_delta()
    model.eval()
    summary = model.trainable_parameter_summary()
    mask_norm = model.mask_emb.detach().norm().item()
    print('conversion done in {:.1f}s; trainable summary: {}'.format(time.time() - t0, summary), flush=True)
    print('  lora={:.2f}M conv={:.2f}M mask_emb={} other={}; |mask_emb|={:.4f}'.format(
        summary['lora'] / 1e6, summary['conv'] / 1e6, summary['mask_emb'], summary['other'], mask_norm), flush=True)
    ok = (
        abs(summary['lora'] - EXPECTED_LORA) <= 0.02 * EXPECTED_LORA
        and abs(summary['conv'] - EXPECTED_CONV) <= 0.02 * EXPECTED_CONV
        and summary['mask_emb'] == 896
        and summary['other'] == 0
        and math.isfinite(mask_norm) and mask_norm > 0.0
    )
    report(
        'delta-conversion',
        ok,
        'lora={} (expect ~{}), conv={} (expect ~{}), mask_emb={}, other={}, mask_emb initialized (norm {:.4f})'.format(
            summary['lora'], EXPECTED_LORA, summary['conv'], EXPECTED_CONV, summary['mask_emb'], summary['other'], mask_norm),
    )

    # ---- 3. forward loss on a synthetic batch ----
    tokenizer = get_qwen_tokenizer(token_path=QWEN_DIR, skip_special_tokens=True, version='cosyvoice3')
    texts = [
        'The quick brown fox jumps over the lazy dog near the quiet river bank.',
        'Speech synthesis with diffusion models is fun.',
    ]
    text_ids = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
    text_len = torch.tensor([len(i) for i in text_ids], dtype=torch.int32)
    text_token = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
    speech_len = torch.tensor([60, 45], dtype=torch.int32)
    speech_token = torch.randint(0, SPEECH_TOKEN_SIZE, (2, int(speech_len.max())), dtype=torch.long)
    batch = {'text_token': text_token, 'text_token_len': text_len, 'speech_token': speech_token, 'speech_token_len': speech_len}
    print('forward batch: text lens {}, speech lens {}'.format(text_len.tolist(), speech_len.tolist()), flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.forward(batch, torch.device('cpu'))
    loss = float(out['loss'])
    acc = float(out['acc'])
    dt_forward = time.time() - t0
    # diagnostic second forward: force t=1 (every target position masked, 1/t weight == 1)
    # so the returned loss equals the raw unweighted mean CE, comparable to the ~10 ballpark.
    saved = (model.t_min, model.prompt_ratio_max, model.prompt_drop)
    model.t_min, model.prompt_ratio_max, model.prompt_drop = 1.0, 0.0, 0.0
    with torch.no_grad():
        raw_ce = float(model.forward(batch, torch.device('cpu'))['loss'])
    model.t_min, model.prompt_ratio_max, model.prompt_drop = saved
    print('forward: loss={:.4f} (1/t weighted) raw_ce={:.4f} (t=1 diagnostic) acc={:.4f} ({:.1f}s)'.format(loss, raw_ce, acc, dt_forward), flush=True)
    if raw_ce > 15.0:
        print('  WARNING raw CE {:.4f} above the ~10 ballpark expected from the frozen AR backbone'.format(raw_ce), flush=True)
    report(
        'forward-loss',
        math.isfinite(loss) and math.isfinite(raw_ce),
        'weighted loss={:.4f} (finite; 1/t weighting inflates small-t draws), raw CE={:.4f} (expected around <=10), acc={:.4f}, {:.1f}s'.format(loss, raw_ce, acc, dt_forward),
    )

    # ---- 4. diffusion decode ----
    schedule = unmask_schedule(TARGET_LEN, NUM_STEPS, model.mu)
    print('unmask schedule (L={}, T={}, mu={}): {} (sum={})'.format(TARGET_LEN, NUM_STEPS, model.mu, schedule, sum(schedule)), flush=True)
    text = torch.tensor([tokenizer.encode('Hello there, this is a quick smoke test sentence.')], dtype=torch.long)
    prompt_text = torch.tensor([tokenizer.encode('A short prompt.')], dtype=torch.long)
    prompt_speech_token = torch.randint(0, SPEECH_TOKEN_SIZE, (1, PROMPT_SPEECH_LEN), dtype=torch.long)
    t0 = time.time()
    tokens = list(model.inference_diffusion(
        text=text,
        text_len=torch.tensor([text.size(1)], dtype=torch.int32),
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([prompt_text.size(1)], dtype=torch.int32),
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=torch.tensor([PROMPT_SPEECH_LEN], dtype=torch.int32),
        embedding=torch.zeros(1, 192),
        num_steps=NUM_STEPS,
        target_len=TARGET_LEN,
    ))
    dt_decode = time.time() - t0
    in_range = all(0 <= t < SPEECH_TOKEN_SIZE for t in tokens)
    print('decoded {} tokens in {:.1f}s ({:.1f}s/step): {}'.format(len(tokens), dt_decode, dt_decode / NUM_STEPS, tokens), flush=True)
    report(
        'diffusion-decode',
        len(tokens) == TARGET_LEN and in_range and sum(schedule) == TARGET_LEN,
        'yielded {} (expect {}), all in [0, {}): {}, schedule sums to L: {}, {:.1f}s total'.format(
            len(tokens), TARGET_LEN, SPEECH_TOKEN_SIZE, in_range, sum(schedule) == TARGET_LEN, dt_decode),
    )

    print('-' * 72, flush=True)
    n_fail = sum(1 for _, p in RESULTS if not p)
    for name, passed in RESULTS:
        print('  {}  {}'.format('PASS' if passed else 'FAIL', name), flush=True)
    print('OVERALL: {}'.format('PASS' if n_fail == 0 else 'FAIL ({} failed)'.format(n_fail)), flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
