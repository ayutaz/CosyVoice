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
import logging
import queue
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper

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
    # Convert audio to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        # NOTE each session owns a CUDA stream, taking one from the pool lets utterances
        # run concurrently on the GPU while keeping exact batch-1 numerics
        session = session_pool.get()
        try:
            speech_token = session.run(None, {session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                              session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        finally:
            session_pool.put(session)
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    for future in tqdm(as_completed(all_task)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    parser.add_argument("--num_sessions", type=int, default=1,
                        help="parallel onnx sessions (one CUDA stream each, ~1-2GB VRAM per session); "
                             "1 keeps the original single-session behavior")
    args = parser.parse_args()

    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir), encoding='utf-8') as f:
        for l in f:
            l = l.replace('\n', '').split(maxsplit=1)
            utt2wav[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider"]
    session_pool = queue.Queue()
    for _ in range(max(args.num_sessions, 1)):
        session_pool.put(onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers))
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
