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
"""DELTA-TTS speed/quality benchmark: AR baseline vs diffusion decoding at several T.

Measures, on the same machine with the same sentences and seeds:
  quality : whisper CER, exact-match (CER==0) count, campplus speaker similarity
            (cosine between the prompt wav embedding and each generated wav)
  speed   : end-to-end RTF (wall / generated seconds), LM-only RTF (the token
            generator is timed inside llm_job, flow/hift excluded), speedup vs
            the AR path binned by the AR audio length (paper table-3 style)

Each path gets one untimed warmup synthesis before measurement (first-call CUDA
warmup would otherwise inflate the AR path by ~40%). The delta conversion is
irreversible in-process, so the AR path always runs first.

Usage:
  python scripts/eval_delta_speed_quality.py \
      --delta_checkpoint checkpoints/delta_ja/avg_20k30k_delta.pt \
      --num_steps 16 8 4 --out_dir eval_out_speed_quality
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'Matcha-TTS'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from eval_ja_cer import SENTENCES, cer  # noqa: E402


def timed_generator(bound_method, sink):
    """Wrap a token-generator method so the wall time of full consumption lands in sink."""
    def wrapped(*args, **kwargs):
        gen = bound_method(*args, **kwargs)

        def run():
            start = time.perf_counter()
            for token in gen:
                yield token
            sink.append(time.perf_counter() - start)
        return run()
    return wrapped


def synthesize_path(cosyvoice, name, sentences, prompt_wav, out_dir, instruct_prefix, seed, lm_method_name):
    from cosyvoice.utils.file_utils import audio_save
    path_dir = os.path.join(out_dir, name)
    os.makedirs(path_dir, exist_ok=True)
    lm_times = []
    llm = cosyvoice.model.llm
    setattr(llm, lm_method_name, timed_generator(getattr(llm, lm_method_name), lm_times))
    rows = []
    try:
        for i, sentence in enumerate([sentences[0]] + list(sentences)):  # index 0 = untimed warmup
            warmup = i == 0
            torch.manual_seed(seed + i - (0 if warmup else 1))
            import random as _random
            _random.seed(seed + i - (0 if warmup else 1))
            start = time.perf_counter()
            chunks = [j['tts_speech'] for j in cosyvoice.inference_cross_lingual(
                instruct_prefix + sentence, prompt_wav, stream=False, text_frontend=False)]
            wall = time.perf_counter() - start
            speech = torch.concat(chunks, dim=1) if chunks else torch.zeros(1, 0)
            if warmup:
                lm_times.clear()
                continue
            wav = os.path.join(path_dir, '{:03d}.wav'.format(i - 1))
            audio_save(wav, speech, cosyvoice.sample_rate)
            rows.append({
                'idx': i - 1, 'ref': sentence, 'wav': wav,
                'audio_sec': speech.shape[1] / cosyvoice.sample_rate,
                'wall_sec': wall,
                'lm_sec': lm_times[-1] if lm_times else None,
            })
    finally:
        delattr(llm, lm_method_name)  # restore the class method
    return rows


def campplus_embedding(session, wav_path):
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from cosyvoice.utils.file_utils import audio_load
    audio, sample_rate = audio_load(wav_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
    return torch.tensor(emb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--ft_llm', default='checkpoints/cosyvoice3_ja/llm.pt')
    parser.add_argument('--delta_checkpoint', required=True)
    parser.add_argument('--prompt_wav', default='./asset/zero_shot_prompt.wav')
    parser.add_argument('--out_dir', default='eval_out_speed_quality')
    parser.add_argument('--whisper_model', default='large-v3')
    parser.add_argument('--num_sentences', type=int, default=len(SENTENCES))
    parser.add_argument('--num_steps', type=int, nargs='+', default=[16, 8, 4])
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()
    sentences = SENTENCES[:args.num_sentences]
    instruct_prefix = 'You are a helpful assistant.<|endofprompt|>'
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. synthesis: AR first (delta conversion consumes the AR llm)
    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice = AutoModel(model_dir=args.model_dir)
    state = torch.load(args.ft_llm, map_location='cpu', weights_only=True)
    cosyvoice.model.llm.load_state_dict({k: v for k, v in state.items() if k not in ('epoch', 'step')}, strict=True)
    cosyvoice.model.llm.to(cosyvoice.model.device).eval()
    results = {}
    print('=== synthesizing ar (ft_kanji) ===')
    results['ar'] = synthesize_path(cosyvoice, 'ar', sentences, args.prompt_wav,
                                    args.out_dir, instruct_prefix, args.seed, 'inference')

    from cosyvoice.llm.diffusion_llm import DiffusionCosyVoice3LM
    cosyvoice.model.llm = DiffusionCosyVoice3LM.from_ar(cosyvoice.model.llm, delta_checkpoint=args.delta_checkpoint)
    cosyvoice.model.llm.to(cosyvoice.model.device).eval()
    for T in args.num_steps:
        name = 'delta_T{}'.format(T)
        print('=== synthesizing', name, '===')
        cosyvoice.model.llm.num_steps = T
        results[name] = synthesize_path(cosyvoice, name, sentences, args.prompt_wav,
                                        args.out_dir, instruct_prefix, args.seed, 'inference_diffusion')
    cosyvoice = None
    torch.cuda.empty_cache()

    # 2. speaker similarity (campplus, prompt vs generated)
    import onnxruntime
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3
    session = onnxruntime.InferenceSession(os.path.join(args.model_dir, 'campplus.onnx'), sess_options=opts,
                                           providers=['CPUExecutionProvider'])
    prompt_emb = campplus_embedding(session, args.prompt_wav)
    for rows in results.values():
        for row in rows:
            emb = campplus_embedding(session, row['wav'])
            row['sim'] = float(torch.nn.functional.cosine_similarity(prompt_emb, emb, dim=0))

    # 3. whisper CER
    import whisper
    print('loading whisper', args.whisper_model)
    asr = whisper.load_model(args.whisper_model)
    for rows in results.values():
        for row in rows:
            row['hyp'] = asr.transcribe(row['wav'], language='ja', temperature=0.0)['text']
            row['cer'] = cer(row['ref'], row['hyp'])

    # 4. aggregate: overall + AR-audio-length bins (paper table-3 style)
    ar_len = {r['idx']: r['audio_sec'] for r in results['ar']}
    bins = [(0.0, 4.0), (4.0, 6.0), (6.0, 1e9)]
    summary = {}
    for name, rows in results.items():
        n = len(rows)
        agg = {
            'avg_cer': sum(r['cer'] for r in rows) / n,
            'exact_match': sum(1 for r in rows if r['cer'] == 0.0),
            'avg_sim': sum(r['sim'] for r in rows) / n,
            'avg_rtf': sum(r['wall_sec'] / r['audio_sec'] for r in rows) / n,
            'avg_lm_rtf': sum(r['lm_sec'] / r['audio_sec'] for r in rows) / n,
            'avg_audio_sec': sum(r['audio_sec'] for r in rows) / n,
        }
        if name != 'ar':
            ar_rows = {r['idx']: r for r in results['ar']}
            agg['speedup_e2e'] = (sum(a['wall_sec'] / a['audio_sec'] for a in results['ar']) /
                                  sum(r['wall_sec'] / r['audio_sec'] for r in rows))
            agg['speedup_lm'] = (sum(a['lm_sec'] / a['audio_sec'] for a in results['ar']) /
                                 sum(r['lm_sec'] / r['audio_sec'] for r in rows))
            agg['speedup_by_bin'] = {}
            for lo, hi in bins:
                idxs = [i for i, sec in ar_len.items() if lo <= sec < hi]
                if not idxs:
                    continue
                ar_rtf = sum(ar_rows[i]['wall_sec'] / ar_rows[i]['audio_sec'] for i in idxs) / len(idxs)
                d_rtf = sum(r['wall_sec'] / r['audio_sec'] for r in rows if r['idx'] in idxs) / len(idxs)
                agg['speedup_by_bin']['{:.0f}-{:.0f}s(n={})'.format(lo, min(hi, 99), len(idxs))] = ar_rtf / d_rtf
        summary[name] = agg

    report = {'summary': summary, 'rows': results,
              'config': {'delta_checkpoint': args.delta_checkpoint, 'num_steps': args.num_steps,
                         'num_sentences': args.num_sentences, 'seed': args.seed,
                         'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}}
    with open(os.path.join(args.out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n=== SUMMARY ===')
    header = '{:10s} {:>8s} {:>6s} {:>7s} {:>8s} {:>8s} {:>8s} {:>8s}'.format(
        'path', 'CER', 'exact', 'SIM', 'RTF', 'LM-RTF', 'x-e2e', 'x-LM')
    print(header)
    for name, a in summary.items():
        print('{:10s} {:8.4f} {:>5d}/{:d} {:7.3f} {:8.3f} {:8.3f} {:>8s} {:>8s}'.format(
            name, a['avg_cer'], a['exact_match'], args.num_sentences, a['avg_sim'], a['avg_rtf'], a['avg_lm_rtf'],
            '{:.2f}'.format(a.get('speedup_e2e', 1.0)), '{:.2f}'.format(a.get('speedup_lm', 1.0))))
    print('report saved to', os.path.join(args.out_dir, 'report.json'))


if __name__ == '__main__':
    main()
