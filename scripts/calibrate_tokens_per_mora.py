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
"""Calibrate tokens_per_mora for the mora-based DELTA-TTS target length rule.

Reads waveform-free training parquet shards (text + speech_token columns) from the
moe_speech delta data archive, computes ratio = len(speech_token) / mora_count(text)
per utterance and reports the distribution (speech tokens are 25Hz). The MEDIAN is
the adopted constant for DiffusionCosyVoice3LM.tokens_per_mora. p60 (=5.0) was
tried as headroom on the theory that under-budget truncation is irrecoverable while
surplus is absorbed by the llm_job silent-token trim, but it regressed BOTH eval
sets (std20 0.187 / holdout 0.360 vs the median's 0.134 / 0.164): fixed-length
decoding is budget-sensitive in both directions, surplus canvas gets filled with
hallucinated speech, not silence. The mean is not used: it is inflated by a right
tail from leading/trailing silence in clips.

Shards inside the archive are named parquet_%09d.tar but are plain parquet files
(CosyVoice convention, see cosyvoice/dataset/processor.py parquet_opener); they are
extracted to a work dir and read directly with pyarrow. Shards are picked evenly
spaced across the archive so the sample is not biased to one recording batch, and
rows are consumed round-robin (one batch per shard per turn) so shards larger than
ROWS_PER_SHARD_HINT still contribute evenly instead of the first shards filling
the whole budget.

Usage:
  python scripts/calibrate_tokens_per_mora.py \
      --archive checkpoints/delta_ja/delta_data_artifacts.tgz \
      --num_utts 20000 --out_json scripts/calibration_tokens_per_mora.json
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import tarfile
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pyarrow.parquet as pq

try:
    from cosyvoice.utils.ja_frontend import mora_count
    MORA_COUNT_SOURCE = 'cosyvoice.utils.ja_frontend.mora_count'
except ImportError:
    # stand-in with the contract rules until ja_frontend.mora_count lands: normal kana = 1,
    # small youon = 0, sokuon / choon / N = 1, space / punctuation = 0, instruct prefix before
    # <|endofprompt|> ignored, non-japanese text = 0 (pyopenjtalk spells Latin letter-by-letter,
    # so the contains_japanese gate is required)
    from cosyvoice.utils.ja_frontend import contains_japanese, ja_text_to_katakana
    MORA_COUNT_SOURCE = 'local fallback (ja_frontend.mora_count not implemented yet)'
    _SMALL_KANA = set('ャュョァィゥェォヮ')

    def mora_count(text):
        text = text.split('<|endofprompt|>')[-1]
        if not contains_japanese(text):
            return 0
        reading = ja_text_to_katakana(text)
        return sum(1 for ch in reading if ch not in _SMALL_KANA and (ch == 'ー' or 0x30A1 <= ord(ch) <= 0x30FA))

_SHARD_RE = re.compile(r'data/train/parquet/parquet_\d+\.tar$')
ROWS_PER_SHARD_HINT = 1000  # tools/make_parquet_list.py --num_utts_per_parquet, only used to pick the shard count
# ratios above this are broken alignments (>= 0.8s of audio per mora, silence-padded or misclipped)
RATIO_MAX = 20.0


def percentile(sorted_vals, p):
    # linear interpolation, p in [0, 100]
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def extract_shard(tf, member, work_dir):
    # flatten to basename, shards are read as plain parquet files despite the .tar name
    path = os.path.join(work_dir, os.path.basename(member.name))
    if not (os.path.isfile(path) and os.path.getsize(path) == member.size):
        with tf.extractfile(member) as src, open(path, 'wb') as dst:
            dst.write(src.read())
    return path


def parse_args():
    parser = argparse.ArgumentParser(description='calibrate tokens_per_mora from moe_speech delta training parquets')
    parser.add_argument('--archive', default=os.path.join(REPO_ROOT, 'checkpoints', 'delta_ja', 'delta_data_artifacts.tgz'),
                        help='tgz with data/train/parquet/parquet_*.tar shards')
    parser.add_argument('--num_utts', type=int, default=20000, help='number of utterances to scan')
    parser.add_argument('--out_json', default=os.path.join(REPO_ROOT, 'scripts', 'calibration_tokens_per_mora.json'),
                        help='where to write the calibration result')
    parser.add_argument('--work_dir', default=None, help='shard extraction dir, reused across runs (default: fresh temp dir)')
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    work_dir = args.work_dir or tempfile.mkdtemp(prefix='tokens_per_mora_')
    os.makedirs(work_dir, exist_ok=True)

    ratios, used_rows = [], {}  # used_rows: shard basename -> rows actually consumed
    scanned = excluded_mora0 = excluded_ratio = 0

    def consume(batch, shard):
        nonlocal scanned, excluded_mora0, excluded_ratio
        for text, speech_token in zip(batch.column('text').to_pylist(), batch.column('speech_token').to_pylist()):
            if scanned >= args.num_utts:
                return
            scanned += 1
            used_rows[shard] = used_rows.get(shard, 0) + 1
            mora = mora_count(text)
            if mora == 0:
                excluded_mora0 += 1
                continue
            ratio = len(speech_token) / mora
            if ratio > RATIO_MAX:
                excluded_ratio += 1
                continue
            ratios.append(ratio)

    def batches(path):
        return pq.ParquetFile(path).iter_batches(batch_size=256, columns=['text', 'speech_token'])

    with tarfile.open(args.archive, 'r:gz') as tf:
        members = {m.name: m for m in tf.getmembers() if _SHARD_RE.search(m.name)}
        names = sorted(members)
        assert len(names) > 0, 'no parquet shards found in {}'.format(args.archive)
        needed = min(len(names), max(1, math.ceil(args.num_utts / ROWS_PER_SHARD_HINT)))
        picked = sorted({names[int(i * len(names) / needed)] for i in range(needed)})
        spare = [n for n in names if n not in set(picked)]
        # extract in archive offset order, gzip tarfile only seeks forward cheaply
        for name in sorted(picked, key=lambda n: members[n].offset):
            extract_shard(tf, members[name], work_dir)
        # round-robin one batch per shard per turn: shards holding more rows than
        # ROWS_PER_SHARD_HINT must not let the first shards fill the whole budget
        readers = [(os.path.basename(n), batches(os.path.join(work_dir, os.path.basename(n)))) for n in picked]
        while scanned < args.num_utts and len(readers) > 0:
            alive = []
            for shard, it in readers:
                if scanned >= args.num_utts:
                    break
                batch = next(it, None)
                if batch is None:
                    continue
                consume(batch, shard)
                alive.append((shard, it))
            readers = alive
        # top up when shards were shorter than the hint
        while scanned < args.num_utts and len(spare) > 0:
            name = spare.pop(0)
            path = extract_shard(tf, members[name], work_dir)
            for batch in batches(path):
                if scanned >= args.num_utts:
                    break
                consume(batch, os.path.basename(name))

    assert len(ratios) > 0, 'no usable utterances (mora_count returned 0 everywhere?)'
    ratios.sort()
    result = {
        'tokens_per_mora': round(statistics.median(ratios), 3),  # adopted constant: the median, see module docstring
        'n': len(ratios),
        'mean': statistics.fmean(ratios),
        'median': statistics.median(ratios),
        'p10': percentile(ratios, 10),
        'p25': percentile(ratios, 25),
        'p60': percentile(ratios, 60),
        'p70': percentile(ratios, 70),
        'p75': percentile(ratios, 75),
        'p90': percentile(ratios, 90),
        'min': ratios[0],
        'max': ratios[-1],
        'scanned_utts': scanned,
        'excluded_mora0': excluded_mora0,
        'excluded_ratio_gt_max': excluded_ratio,
        'ratio_max': RATIO_MAX,
        'archive': os.path.relpath(args.archive, REPO_ROOT).replace(os.sep, '/'),
        'shards_used': sorted(used_rows),  # only shards that actually contributed rows
        'shard_rows': {k: used_rows[k] for k in sorted(used_rows)},
        'shards_in_archive': len(names),
        'mora_count_source': MORA_COUNT_SOURCE,
        'token_frame_rate_hz': 25,
        'date': time.strftime('%Y-%m-%d'),
        'elapsed_sec': round(time.time() - start, 1),
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
