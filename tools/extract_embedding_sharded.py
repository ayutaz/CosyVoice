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
"""Run tools/extract_embedding.py over N wav.scp shards in parallel processes and merge.

Single-process throughput is GIL-bound (~16 utts/s regardless of thread count) and
batch-1 GPU inference of the tiny campplus model is launch-overhead-bound, so on a
many-core box independent CPU processes are the fastest option (measured 577 utts/s
with 14 processes on 128 cores vs 16 utts/s single-process). spk2embedding is
recomputed globally after the merge because speakers may span shards.
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
    parser.add_argument('--num_procs', type=int, default=16)
    parser.add_argument('--num_thread', type=int, default=8,
                        help='threads per worker process')
    args = parser.parse_args()

    with open('{}/wav.scp'.format(args.dir), encoding='utf-8') as f:
        lines = f.read().splitlines()
    with open('{}/utt2spk'.format(args.dir), encoding='utf-8') as f:
        utt2spk = dict(l.split(maxsplit=1) for l in f.read().splitlines())

    shard_dirs = []
    for i in range(args.num_procs):
        shard = lines[i::args.num_procs]
        if not shard:
            continue
        sd = '{}/shard_{}'.format(args.dir, i)
        os.makedirs(sd, exist_ok=True)
        with open('{}/wav.scp'.format(sd), 'w', encoding='utf-8') as f:
            f.write('\n'.join(shard) + '\n')
        with open('{}/utt2spk'.format(sd), 'w', encoding='utf-8') as f:
            for l in shard:
                utt = l.split(maxsplit=1)[0]
                f.write('{} {}\n'.format(utt, utt2spk[utt]))
        shard_dirs.append(sd)

    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_embedding.py')
    procs = []
    for sd in shard_dirs:
        log = open('{}/extract.log'.format(sd), 'w')
        procs.append((sd, subprocess.Popen(
            [sys.executable, tool, '--dir', sd, '--onnx_path', args.onnx_path,
             '--num_thread', str(args.num_thread)],
            stdout=log, stderr=subprocess.STDOUT)))
    failed = []
    for sd, p in procs:
        if p.wait() != 0:
            failed.append(sd)
    if failed:
        print('FAILED shards:', failed)
        sys.exit(1)

    utt2embedding = {}
    for sd in shard_dirs:
        utt2embedding.update(torch.load('{}/utt2embedding.pt'.format(sd), weights_only=True))
    assert len(utt2embedding) == len(lines), (len(utt2embedding), len(lines))
    spk2sum = {}
    for utt, emb in utt2embedding.items():
        spk2sum.setdefault(utt2spk[utt], []).append(emb)
    spk2embedding = {spk: torch.tensor(v).mean(dim=0).tolist() for spk, v in spk2sum.items()}
    torch.save(utt2embedding, '{}/utt2embedding.pt'.format(args.dir))
    torch.save(spk2embedding, '{}/spk2embedding.pt'.format(args.dir))
    print('MERGE_DONE utts={} spks={}'.format(len(utt2embedding), len(spk2embedding)))


if __name__ == '__main__':
    main()
