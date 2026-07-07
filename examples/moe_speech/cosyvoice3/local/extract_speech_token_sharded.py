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
"""Run extract_speech_token_batch.py over N wav.scp shards in parallel processes and merge.

A single process is GIL-bound on the mel producers regardless of thread count, so the
GPU sits mostly idle. N independent processes each own an ONNX CUDA session (its own
stream) and a producer pool; equal-length batching keeps the numerics of the batch-1
reference. Same pattern as tools/extract_embedding_sharded.py.
"""
import argparse
import os
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--onnx_path', required=True)
    parser.add_argument('--num_procs', type=int, default=8)
    parser.add_argument('--num_thread', type=int, default=8,
                        help='producer threads per worker process')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--verify_num', type=int, default=0,
                        help='per-shard batch-1 cross-check count')
    args = parser.parse_args()

    with open('{}/wav.scp'.format(args.dir), encoding='utf-8') as f:
        lines = f.read().splitlines()

    shard_dirs = []
    for i in range(args.num_procs):
        shard = lines[i::args.num_procs]
        if not shard:
            continue
        sd = '{}/tok_shard_{}'.format(args.dir, i)
        os.makedirs(sd, exist_ok=True)
        with open('{}/wav.scp'.format(sd), 'w', encoding='utf-8') as f:
            f.write('\n'.join(shard) + '\n')
        shard_dirs.append(sd)

    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_speech_token_batch.py')
    procs = []
    for sd in shard_dirs:
        log = open('{}/extract.log'.format(sd), 'w')
        procs.append((sd, subprocess.Popen(
            [sys.executable, tool, '--dir', sd, '--onnx_path', args.onnx_path,
             '--num_thread', str(args.num_thread), '--batch_size', str(args.batch_size),
             '--verify_num', str(args.verify_num)],
            stdout=log, stderr=subprocess.STDOUT)))
    failed = []
    for sd, p in procs:
        if p.wait() != 0:
            failed.append(sd)
    if failed:
        print('FAILED shards:', failed)
        sys.exit(1)

    utt2speech_token = {}
    for sd in shard_dirs:
        utt2speech_token.update(torch.load('{}/utt2speech_token.pt'.format(sd), weights_only=True))
    assert len(utt2speech_token) == len(lines), (len(utt2speech_token), len(lines))
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))
    print('MERGE_DONE utts={}'.format(len(utt2speech_token)))


if __name__ == '__main__':
    main()
