#!/usr/bin/env python3
# Copyright (c) 2026 Alibaba Inc
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
"""S9: end-to-end smoke of the DELTA-TTS CLI wiring (frontend -> diffusion LM -> flow -> hift).

Runs one Japanese sentence through the full CosyVoice3 pipeline twice: first with the
AR llm (pretrained, or the Japanese fine-tune when checkpoints/cosyvoice3_ja/llm.pt
exists) as a timing/behavior reference, then after DiffusionCosyVoice3LM.from_ar()
conversion through the same inference_cross_lingual entry point, exercising the
CosyVoice3Model.llm_job dispatch to inference_diffusion.

The delta conversion is UNTRAINED here (zero-init LoRA-B/conv), so the delta audio is
expected to be degenerate; this validates mechanics and wiring only, not quality.
Timings are indicative only (different generated lengths, tiny sample).

Usage:
  .venv/Scripts/python.exe scripts/spikes/s9_delta_e2e_smoke.py [--num_steps 4] [--out_dir <dir>]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'third_party', 'Matcha-TTS'))

import torch

TEXT = 'You are a helpful assistant.<|endofprompt|>今日は良い天気なので、公園まで散歩に行きました。'


def synth(cosyvoice, prompt_wav, label):
    torch.manual_seed(1986)
    import random as _random
    _random.seed(1986)
    start = time.time()
    chunks = [j['tts_speech'] for j in cosyvoice.inference_cross_lingual(
        TEXT, prompt_wav, stream=False, text_frontend=False)]
    elapsed = time.time() - start
    speech = torch.concat(chunks, dim=1) if chunks else torch.zeros(1, 0)
    sec = speech.shape[1] / cosyvoice.sample_rate
    ok = speech.shape[1] > 0 and bool(torch.isfinite(speech).all())
    # NOTE keep prints ASCII: the Windows console is cp932 here
    print('[{}] {} : audio {:.2f}s, wall {:.2f}s, rtf {:.3f}'.format(
        'PASS' if ok else 'FAIL', label, sec, elapsed, elapsed / sec if sec > 0 else float('inf')))
    return ok, speech, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--ja_llm', default='checkpoints/cosyvoice3_ja/llm.pt')
    parser.add_argument('--prompt_wav', default='./asset/zero_shot_prompt.wav')
    parser.add_argument('--num_steps', type=int, default=4, help='diffusion steps for the smoke (16 = paper)')
    parser.add_argument('--out_dir', default=os.environ.get('S9_OUT_DIR', 'eval_out/s9_smoke'))
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.file_utils import audio_save
    from cosyvoice.llm.diffusion_llm import DiffusionCosyVoice3LM

    os.makedirs(args.out_dir, exist_ok=True)
    cosyvoice = AutoModel(model_dir=args.model_dir)
    if os.path.exists(args.ja_llm):
        state = torch.load(args.ja_llm, map_location='cpu', weights_only=True)
        state = {k: v for k, v in state.items() if k not in ('epoch', 'step')}
        cosyvoice.model.llm.load_state_dict(state, strict=True)
        cosyvoice.model.llm.to(cosyvoice.model.device).eval()
        print('AR backbone: Japanese fine-tune', args.ja_llm)
    else:
        print('AR backbone: pretrained (no', args.ja_llm, 'found)')

    results = {}
    results['ar'], speech, _ = synth(cosyvoice, args.prompt_wav, 'AR reference')
    audio_save(os.path.join(args.out_dir, 'ar.wav'), speech, cosyvoice.sample_rate)

    cosyvoice.model.llm = DiffusionCosyVoice3LM.from_ar(cosyvoice.model.llm, num_steps=args.num_steps)
    cosyvoice.model.llm.to(cosyvoice.model.device).eval()
    summary = cosyvoice.model.llm.trainable_parameter_summary()
    counts_ok = summary == {'lora': 35192832, 'conv': 58641408, 'mask_emb': 896, 'other': 0}
    print('[{}] conversion : trainable {}'.format('PASS' if counts_ok else 'FAIL', summary))
    results['conversion'] = counts_ok

    results['delta'], speech, _ = synth(cosyvoice, args.prompt_wav, 'delta T={} (untrained)'.format(args.num_steps))
    audio_save(os.path.join(args.out_dir, 'delta_untrained.wav'), speech, cosyvoice.sample_rate)

    print('OVERALL', 'PASS' if all(results.values()) else 'FAIL', results)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
