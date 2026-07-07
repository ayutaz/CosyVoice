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
"""Prepare the moe-speech-plus dataset (https://huggingface.co/datasets/ayousanz/moe-speech-plus)
for CosyVoice3 training.

The dataset ships one zip per character ({uuid}.zip) containing
data/{uuid}/wav/{uuid}_NNN.wav plus a metadata json next to each wav with two ASR
transcriptions (parakeet / anime-whisper), duration and speechMOS. Text is kept as raw
kanji-mixed japanese so the model learns kanji readings. Utterances are filtered by
speechMOS and by the agreement (CER) between the two transcriptions, since both are
ASR pseudo labels.
"""
import argparse
import collections
import json
import logging
import os
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

TRANSCRIPTION_KEYS = {
    'parakeet': 'parakeet_jp_transcription',
    'anime_whisper': 'anime_whisper_transcription',
}
# punctuation/style characters ignored when comparing the two transcriptions
_STYLE_CHARS = '、。！？!?…‥・♪〜~ー――「」『』（）() 　.,'
_STYLE_TABLE = str.maketrans('', '', _STYLE_CHARS)


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


def transcription_cer(a, b):
    """CER between two transcriptions after dropping punctuation/style characters."""
    a, b = a.translate(_STYLE_TABLE), b.translate(_STYLE_TABLE)
    if a == b:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    return edit_distance(a, b) / max(len(a), len(b))


def filter_utt(meta, transcription='parakeet', mos_threshold=2.5, max_cross_cer=0.2,
               min_duration=1.0, max_duration=29.0):
    """Return (text, None) when the utterance is usable, else (None, reject reason)."""
    text = (meta.get(TRANSCRIPTION_KEYS[transcription]) or '').strip().replace('\n', ' ')
    if text == '':
        return None, 'empty_text'
    duration = meta.get('duration')
    if duration is not None and not min_duration <= duration <= max_duration:
        return None, 'duration'
    mos = meta.get('speechMOS')
    if mos is not None and mos < mos_threshold:
        return None, 'mos'
    if max_cross_cer >= 0:
        other_key = [v for k, v in TRANSCRIPTION_KEYS.items() if k != transcription][0]
        other = (meta.get(other_key) or '').strip()
        if other != '' and transcription_cer(text, other) > max_cross_cer:
            return None, 'cross_cer'
    return text, None


def find_meta_path(wav):
    candidates = [
        wav[:-len('.wav')] + '.json',
        wav[:-len('.wav')].replace('{}wav{}'.format(os.sep, os.sep), '{}json{}'.format(os.sep, os.sep)) + '.json',
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def extract_zip(zip_path, extract_dir):
    done_marker = os.path.join(extract_dir, '.done_{}'.format(os.path.basename(zip_path).replace('.zip', '')))
    if os.path.exists(done_marker):
        return zip_path, True
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    open(done_marker, 'w').close()
    return zip_path, False


def write_kaldi_dir(des_dir, utts, utt2wav, utt2text, utt2spk, instruct):
    os.makedirs(des_dir, exist_ok=True)
    spk2utt = collections.defaultdict(list)
    for utt in utts:
        spk2utt[utt2spk[utt]].append(utt)
    with open('{}/wav.scp'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {}\n'.format(utt, utt2wav[utt]))
    with open('{}/text'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {}\n'.format(utt, utt2text[utt]))
    with open('{}/utt2spk'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {}\n'.format(utt, utt2spk[utt]))
    with open('{}/spk2utt'.format(des_dir), 'w', encoding='utf-8') as f:
        for spk, spk_utts in spk2utt.items():
            f.write('{} {}\n'.format(spk, ' '.join(spk_utts)))
    if instruct != '':
        with open('{}/instruct'.format(des_dir), 'w', encoding='utf-8') as f:
            for utt in utts:
                f.write('{} {}\n'.format(utt, instruct))


def main():
    zips = sorted(glob('{}/*.zip'.format(args.src_dir)))
    if len(zips) > 0:
        os.makedirs(args.extract_dir, exist_ok=True)
        logger.info('extracting {} zips to {}'.format(len(zips), args.extract_dir))
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            tasks = [executor.submit(extract_zip, z, args.extract_dir) for z in zips]
            for task in tqdm(as_completed(tasks), total=len(tasks)):
                task.result()
        search_dir = args.extract_dir
    else:
        logger.info('no zip found in {}, assume it is already extracted'.format(args.src_dir))
        search_dir = args.src_dir

    wavs = sorted(glob('{}/**/*.wav'.format(search_dir), recursive=True))
    logger.info('found {} wavs'.format(len(wavs)))

    utt2wav, utt2text, utt2spk = {}, {}, {}
    rejects = collections.Counter()
    for wav in tqdm(wavs):
        meta_path = find_meta_path(wav)
        if meta_path is None:
            rejects['no_meta'] += 1
            continue
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        text, reason = filter_utt(meta, transcription=args.transcription, mos_threshold=args.mos_threshold,
                                  max_cross_cer=args.max_cross_cer, min_duration=args.min_duration,
                                  max_duration=args.max_duration)
        if text is None:
            rejects[reason] += 1
            continue
        utt = os.path.basename(wav).replace('.wav', '')
        utt2wav[utt] = os.path.abspath(wav)
        utt2text[utt] = text
        utt2spk[utt] = utt.split('_')[0]
    logger.info('kept {} / {} utts, rejects: {}'.format(len(utt2wav), len(wavs), dict(rejects)))

    # hold out dev utterances per speaker for cross validation
    spk2utt = collections.defaultdict(list)
    for utt in utt2wav:
        spk2utt[utt2spk[utt]].append(utt)
    rng = random.Random(args.seed)
    train_utts, dev_utts = [], []
    for spk, spk_utts in sorted(spk2utt.items()):
        spk_utts = sorted(spk_utts)
        rng.shuffle(spk_utts)
        n_dev = min(args.dev_utts_per_spk, max(len(spk_utts) - 1, 0))
        dev_utts.extend(spk_utts[:n_dev])
        train_utts.extend(spk_utts[n_dev:])
    train_utts.sort()
    dev_utts.sort()
    logger.info('train {} utts / dev {} utts / {} spks'.format(len(train_utts), len(dev_utts), len(spk2utt)))

    write_kaldi_dir('{}/train'.format(args.des_dir), train_utts, utt2wav, utt2text, utt2spk, args.instruct)
    write_kaldi_dir('{}/dev'.format(args.des_dir), dev_utts, utt2wav, utt2text, utt2spk, args.instruct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True,
                        help='dir containing {uuid}.zip files downloaded from huggingface')
    parser.add_argument('--extract_dir', type=str, default='',
                        help='where to unzip, default <src_dir>/extracted')
    parser.add_argument('--des_dir', type=str, required=True,
                        help='output dir, train/dev kaldi style subdirs are created')
    parser.add_argument('--transcription', type=str, default='parakeet', choices=list(TRANSCRIPTION_KEYS.keys()))
    parser.add_argument('--mos_threshold', type=float, default=2.5)
    parser.add_argument('--max_cross_cer', type=float, default=0.2,
                        help='max CER between the two ASR transcriptions, -1 to disable')
    parser.add_argument('--min_duration', type=float, default=1.0)
    parser.add_argument('--max_duration', type=float, default=29.0)
    parser.add_argument('--dev_utts_per_spk', type=int, default=1)
    parser.add_argument('--instruct', type=str, default='You are a helpful assistant.<|endofprompt|>')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()
    if args.extract_dir == '':
        args.extract_dir = os.path.join(args.src_dir, 'extracted')
    main()
