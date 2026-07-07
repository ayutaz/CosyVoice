#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

from cosyvoice.utils.file_utils import audio_load

# NOTE constructing torchaudio.transforms.Resample recomputes the sinc kernel, cache one
# transform per source rate instead of building one per utterance
_resamplers = {}


def get_resampler(orig_freq):
    if orig_freq not in _resamplers:
        _resamplers[orig_freq] = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=16000)
    return _resamplers[orig_freq]


def single_job(utt):
    audio, sample_rate = audio_load(utt2wav[utt])
    if sample_rate != 16000:
        audio = get_resampler(sample_rate)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    session = session_pool.get()
    try:
        embedding = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    finally:
        session_pool.put(session)
    return utt, embedding


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2embedding, spk2embedding = {}, {}
    for future in tqdm(as_completed(all_task)):
        utt, embedding = future.result()
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="campplus is small, cpu with many threads is usually enough; "
                             "cuda helps on boxes with few cores")
    parser.add_argument("--num_sessions", type=int, default=1,
                        help="parallel onnx sessions; with cuda each owns its own stream")
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir), encoding='utf-8') as f:
        for l in f:
            l = l.replace('\n', '').split(maxsplit=1)
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir), encoding='utf-8') as f:
        for l in f:
            l = l.replace('\n', '').split(maxsplit=1)
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    # NOTE utterance lengths vary, the default EXHAUSTIVE cudnn algo search re-benchmarks
    # convolutions for every new input shape and slows batch-1 inference to a crawl
    if args.provider == "cuda":
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
    else:
        providers = ["CPUExecutionProvider"]
    session_pool = queue.Queue()
    # NOTE cpu sessions run concurrently from many threads already, extra sessions only
    # pay off with the cuda provider where each session owns its own stream
    for _ in range(max(args.num_sessions, 1)):
        session_pool.put(onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers))
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
