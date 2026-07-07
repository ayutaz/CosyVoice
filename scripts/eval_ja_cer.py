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
"""Japanese CER evaluation: does the finetuned llm read kanji-mixed text directly?

Synthesizes the same sentences through four paths and scores them with whisper ASR
against the reference text:

  base_katakana : pretrained llm  + ja_frontend katakana conversion (phase 1 path)
  base_kanji    : pretrained llm  + raw kanji text  (expected failure baseline)
  ft_katakana   : finetuned llm   + ja_frontend katakana conversion
  ft_kanji      : finetuned llm   + raw kanji text  (the finetuning target)

Usage:
  python scripts/eval_ja_cer.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
      --ft_llm checkpoints/cosyvoice3_ja/llm.pt --out_dir eval_out
"""
import argparse
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'Matcha-TTS'))

import torch

# kanji-mixed sentences covering common failure modes: homograph readings, numbers,
# counters, dates, and ordinary conversational text. Crafted, not from the training set.
SENTENCES = [
    '今日は良い天気なので、公園まで散歩に行きました。',
    '会議は午後三時から始まる予定です。',
    '昨日、銀行で十万円を引き出した。',
    '彼女は東京大学で物理学を研究しています。',
    '新幹線は時速三百キロメートルで走行します。',
    '二千二十四年四月一日に新しい生活が始まった。',
    '一行目の文章を声に出して読んでください。',
    '人気の店には行列ができていた。',
    '風邪を引いたので、薬を飲んで早めに寝ます。',
    '来月の十五日までに書類を提出してください。',
    '図書館で借りた本を三冊返却した。',
    '大人二枚と子供一枚の切符をお願いします。',
    '台風の影響で電車が遅れているようです。',
    '猫が窓際で気持ちよさそうに昼寝をしている。',
    '経済成長率は前年比で二パーセント上昇した。',
    '駅前の交差点を右に曲がると郵便局があります。',
    '彼は毎朝六時に起きて、一時間走っている。',
    '料理の写真を撮ってから食べるのが習慣になった。',
    '週末は家族と温泉旅行に出かける予定です。',
    '説明書をよく読んでから組み立ててください。',
]

# style characters ignored for CER, same spirit as the dataset cross-CER filter
_STRIP_CHARS = '、。！？!?…‥・♪〜~ー――「」『』（）() 　.,　\n'


def normalize_for_cer(text):
    text = unicodedata.normalize('NFKC', text)
    return ''.join(c for c in text if c not in _STRIP_CHARS)


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1]
        for j, cb in enumerate(b):
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(ref, hyp):
    ref, hyp = normalize_for_cer(ref), normalize_for_cer(hyp)
    if len(ref) == 0:
        return 1.0
    return edit_distance(ref, hyp) / len(ref)


def synthesize(cosyvoice, sentences, prompt_wav, out_dir, use_frontend, instruct_prefix, seed):
    from cosyvoice.utils.file_utils import audio_save
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, sentence in enumerate(sentences):
        out_path = os.path.join(out_dir, '{:03d}.wav'.format(i))
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        torch.manual_seed(seed + i)
        import random as _random
        _random.seed(seed + i)
        text = instruct_prefix + sentence
        chunks = [j['tts_speech'] for j in cosyvoice.inference_cross_lingual(
            text, prompt_wav, stream=False, text_frontend=use_frontend)]
        speech = torch.concat(chunks, dim=1) if chunks else torch.zeros(1, 100)
        audio_save(out_path, speech, cosyvoice.sample_rate)
        paths.append(out_path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--ft_llm', default='checkpoints/cosyvoice3_ja/llm.pt')
    parser.add_argument('--prompt_wav', default='./asset/zero_shot_prompt.wav')
    parser.add_argument('--out_dir', default='eval_out')
    parser.add_argument('--whisper_model', default='large-v3')
    parser.add_argument('--num_sentences', type=int, default=len(SENTENCES))
    parser.add_argument('--seed', type=int, default=1986)
    parser.add_argument('--paths', nargs='+',
                        default=['base_katakana', 'base_kanji', 'ft_katakana', 'ft_kanji'])
    args = parser.parse_args()
    sentences = SENTENCES[:args.num_sentences]
    instruct_prefix = 'You are a helpful assistant.<|endofprompt|>'

    # 1. synthesis: base paths first, then swap in the finetuned llm weights
    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice = AutoModel(model_dir=args.model_dir)
    plan = {
        'base_katakana': (False, True),   # (use_ft, use_frontend)
        'base_kanji': (False, False),
        'ft_katakana': (True, True),
        'ft_kanji': (True, False),
    }
    ft_loaded = False
    for name in args.paths:
        use_ft, use_frontend = plan[name]
        if use_ft and not ft_loaded:
            state = torch.load(args.ft_llm, map_location='cpu', weights_only=True)
            state = {k: v for k, v in state.items() if k not in ('epoch', 'step')}
            cosyvoice.model.llm.load_state_dict(state, strict=True)
            cosyvoice.model.llm.to(cosyvoice.model.device).eval()
            ft_loaded = True
            print('loaded finetuned llm from', args.ft_llm)
        print('=== synthesizing', name, '===')
        synthesize(cosyvoice, sentences, args.prompt_wav,
                   os.path.join(args.out_dir, name), use_frontend, instruct_prefix, args.seed)
    del cosyvoice
    torch.cuda.empty_cache()

    # 2. ASR + CER
    import whisper
    print('loading whisper', args.whisper_model)
    asr = whisper.load_model(args.whisper_model)
    report = {}
    for name in args.paths:
        rows = []
        for i, sentence in enumerate(sentences):
            wav = os.path.join(args.out_dir, name, '{:03d}.wav'.format(i))
            hyp = asr.transcribe(wav, language='ja', temperature=0.0)['text']
            rows.append({'ref': sentence, 'hyp': hyp, 'cer': cer(sentence, hyp)})
        avg = sum(r['cer'] for r in rows) / len(rows)
        report[name] = {'avg_cer': avg, 'rows': rows}
        print('{}: avg CER {:.4f}'.format(name, avg))

    with open(os.path.join(args.out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('\n=== SUMMARY ===')
    for name in args.paths:
        print('{:16s} avg CER {:.4f}'.format(name, report[name]['avg_cer']))
    print('report saved to', os.path.join(args.out_dir, 'report.json'))


if __name__ == '__main__':
    main()
