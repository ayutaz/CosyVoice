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
"""Average DELTA-TTS trainable-only checkpoints (train_delta.save_delta_checkpoint format).

SWA-style averaging of nearby training iterates: with the paper's constant lr 1e-4 the
individual step checkpoints oscillate in quality (measured 20-sentence CER on the
japanese run: single step-25000 0.138 vs the 20k+25k+30k average 0.106). Note this
averages lora_A and lora_B separately (mean(A)@mean(B) != mean(A@B)); empirically it
still helps for adjacent iterates, but keep the window narrow.

Usage:
  python scripts/average_delta_checkpoints.py --dst avg_delta.pt \
      --src epoch_1_step_20000_delta.pt epoch_1_step_25000_delta.pt epoch_1_step_30000_delta.pt
"""
import argparse

import torch

BUCKETS = ('lora', 'conv', 'mask_emb')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', required=True, help='output averaged delta checkpoint')
    parser.add_argument('--src', nargs='+', required=True, help='input delta checkpoints')
    args = parser.parse_args()
    states = [torch.load(p, map_location='cpu') for p in args.src]
    for s, p in zip(states, args.src):
        missing = [b for b in BUCKETS if b not in s]
        if missing:
            raise SystemExit('{} is not a delta checkpoint (missing {})'.format(p, missing))
    avg = {'epoch': -1, 'step': max(s.get('step', 0) for s in states)}
    for bucket in BUCKETS:
        keys = set(states[0][bucket])
        for s, p in zip(states, args.src):
            if set(s[bucket]) != keys:
                raise SystemExit('{} has mismatching {} keys'.format(p, bucket))
        avg[bucket] = {k: torch.stack([s[bucket][k].float() for s in states]).mean(0) for k in keys}
    torch.save(avg, args.dst)
    print('averaged {} checkpoints -> {}'.format(len(states), args.dst))


if __name__ == '__main__':
    main()
