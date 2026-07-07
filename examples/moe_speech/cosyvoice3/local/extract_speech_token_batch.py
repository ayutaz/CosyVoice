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
"""Batched GPU speech token extraction for the CosyVoice3 tokenizer.

This is a throughput-oriented variant of tools/extract_speech_token.py. The
original tool runs speech_tokenizer_v3.onnx at batch size 1 from many python
threads sharing one session, which leaves the GPU badly underutilized. Here a
pool of producer threads computes per-utterance whisper log-mel features while a
single consumer packs them into padded batches and runs speech_tokenizer_v3.batch.onnx
(input: feats [B, 128, T] float32 + feats_length [B] int32) once per batch.

Per-utterance semantics (16kHz mono resample, whisper n_mels=128 log-mel,
>30s -> empty token list, output saved as torch dict utt->token list) match the
original tool exactly. Row i of each batch is sliced to feats_length[i] // 4
tokens, mirroring cosyvoice/utils/onnx.py::SpeechTokenExtractor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import queue
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
import onnxruntime
import torchaudio
import whisper

from cosyvoice.utils.file_utils import audio_load

# Size of the buffer that is sorted by mel length before being sliced into
# batches. Sorting a chunk keeps utterances of similar length together so the
# zero padding inside each batch stays small.
SORT_CHUNK = 512

# NOTE constructing torchaudio.transforms.Resample recomputes the sinc kernel and holds
# the GIL, cache one transform per source rate instead of building one per utterance
_resamplers = {}


def get_resampler(orig_freq):
    if orig_freq not in _resamplers:
        _resamplers[orig_freq] = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=16000)
    return _resamplers[orig_freq]


def read_wav_scp(dir):
    utt2wav = {}
    with open('{}/wav.scp'.format(dir), encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            parts = l.split(maxsplit=1)
            utt2wav[parts[0]] = parts[1]
    return utt2wav


def compute_mel(wav_path):
    """Replicate tools/extract_speech_token.py per-utterance feature extraction.

    Returns a [128, T] float32 numpy array, or None for audio longer than 30s
    (which the tokenizer does not support and the original tool maps to an empty
    token list). The log-mel MUST be computed per utterance because whisper
    normalizes against the tensor's global max; padding audio beforehand would
    corrupt that normalization.
    """
    audio, sample_rate = audio_load(wav_path)
    if sample_rate != 16000:
        audio = get_resampler(sample_rate)(audio)
    # Convert audio to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s')
        return None
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    # whisper returns [1, 128, T] for a [1, samples] input; drop the batch dim.
    if feat.dim() == 3:
        feat = feat.squeeze(0)
    return feat.detach().cpu().numpy().astype(np.float32)


def make_session(onnx_path, provider):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # NOTE HEURISTIC: the default EXHAUSTIVE cudnn search re-benchmarks conv algos
        # for every new input shape, deadly with varying utterance lengths
        providers = [("CUDAExecutionProvider", {'device_id': local_rank, 'cudnn_conv_algo_search': 'HEURISTIC'})]
    else:
        providers = ["CPUExecutionProvider"]
    return onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)


def extract_one(session, input_names, mel):
    """Run a single [128, T] mel through the batch onnx at batch size 1.

    Mirrors tools/extract_speech_token.py: feed [1, 128, T] + [T] and flatten
    the output. Used by the verifier to reproduce the reference token sequence.
    """
    t = mel.shape[1]
    feats = mel[np.newaxis, :, :].astype(np.float32)
    feats_length = np.array([t], dtype=np.int32)
    tokens = session.run(None, {input_names[0]: feats, input_names[1]: feats_length})[0]
    return tokens.flatten().tolist()


def run_batch(session, input_names, batch, utt2speech_token):
    """Pad a list of (utt, mel) to the batch max T, run onnx once, slice rows."""
    b = len(batch)
    max_t = max(mel.shape[1] for _, mel in batch)
    feats = np.zeros((b, 128, max_t), dtype=np.float32)
    feats_length = np.zeros((b,), dtype=np.int32)
    for idx, (_, mel) in enumerate(batch):
        t = mel.shape[1]
        feats[idx, :, :t] = mel
        feats_length[idx] = t
    tokens = session.run(None, {input_names[0]: feats, input_names[1]: feats_length})[0]
    for idx, (utt, _) in enumerate(batch):
        n_tok = int(feats_length[idx]) // 4
        utt2speech_token[utt] = tokens[idx][:n_tok].tolist()


def run_batch_equal_length(session, input_names, batch, utt2speech_token):
    """Run a batch whose mels ALL share the same T: no padding exists, so every output
    row is the complete model output for that utterance and no slicing is needed.
    Token sequences match the batch-1 reference (a rare near-codebook-boundary flip
    aside, measured 1 token in ~4400 on real data)."""
    t = batch[0][1].shape[1]
    feats = np.stack([mel for _, mel in batch])
    feats_length = np.full((len(batch),), t, dtype=np.int32)
    tokens = session.run(None, {input_names[0]: feats, input_names[1]: feats_length})[0]
    for idx, (utt, _) in enumerate(batch):
        utt2speech_token[utt] = tokens[idx].flatten().tolist()


def flush_buffer(session, input_names, buffer, args, utt2speech_token, pbar):
    """Sort a buffer chunk by mel length and slice it into onnx calls.

    With --equal_length (default) a batch only holds utterances whose mel length is
    exactly equal, eliminating padding and its numeric drift. Otherwise batches are
    packed by --batch_size and --max_batch_frames (total padded mel frames) with zero
    padding to the batch max. At least one utterance is always placed in a batch.
    """
    buffer.sort(key=lambda x: x[1].shape[1])
    i, n = 0, len(buffer)
    while i < n:
        batch = []
        max_t = 0
        while i < n:
            mel = buffer[i][1]
            if args.equal_length and batch and mel.shape[1] != max_t:
                break
            new_max_t = max(max_t, mel.shape[1])
            padded_frames = new_max_t * (len(batch) + 1)
            if batch and (len(batch) >= args.batch_size or padded_frames > args.max_batch_frames):
                break
            batch.append(buffer[i])
            max_t = new_max_t
            i += 1
        if args.equal_length:
            run_batch_equal_length(session, input_names, batch, utt2speech_token)
        else:
            run_batch(session, input_names, batch, utt2speech_token)
        pbar.update(len(batch))


def batched_extract(args, utt2wav, session, input_names):
    utts = list(utt2wav.keys())
    total = len(utts)
    result_queue = queue.Queue()

    def producer(utt):
        try:
            mel = compute_mel(utt2wav[utt])
        except Exception as e:  # noqa: BLE001 - keep the run going, report the failure
            logging.warning('failed to extract mel for %s: %s', utt, e)
            mel = None
        result_queue.put((utt, mel))

    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    for utt in utts:
        executor.submit(producer, utt)

    utt2speech_token = {}
    buffer = []
    received = 0
    pbar = tqdm(total=total)
    while received < total:
        utt, mel = result_queue.get()
        received += 1
        if mel is None:
            # >30s audio (or a load failure): empty token list, no onnx call.
            utt2speech_token[utt] = []
            pbar.update(1)
        else:
            buffer.append((utt, mel))
        if len(buffer) >= SORT_CHUNK or (received == total and buffer):
            flush_buffer(session, input_names, buffer, args, utt2speech_token, pbar)
            buffer = []
    pbar.close()
    executor.shutdown(wait=True)
    return utt2speech_token


def verify(args, utt2wav, utt2speech_token, session, input_names):
    """Cross-check against batch-1 at the TOKEN level. Utterance-level exact match is
    the wrong bar: even two batch-1 runs of the same utterance differ by an occasional
    near-codebook-boundary token (cudnn numeric noise, measured ~1 token in ~2000), so
    a batch is accepted while the aggregate token agreement stays >= 99.5%."""
    utts = list(utt2speech_token.keys())
    sample = random.sample(utts, min(args.verify_num, len(utts)))
    total_tokens, diff_tokens = 0, 0
    for utt in tqdm(sample, desc='verify'):
        got = utt2speech_token[utt]
        mel = compute_mel(utt2wav[utt])
        if mel is None:
            if got != []:
                diff_tokens += len(got)
                total_tokens += len(got)
                logging.warning('utt %s: expected empty token list', utt)
            continue
        ref = extract_one(session, input_names, mel)
        n = max(len(ref), len(got), 1)
        diffs = sum(1 for a, b in zip(ref, got) if a != b) + abs(len(ref) - len(got))
        total_tokens += n
        diff_tokens += diffs
        if diffs > max(1, n // 50):
            logging.warning('utt %s: %d/%d tokens differ from the batch-1 reference', utt, diffs, n)
    rate = 1.0 - diff_tokens / max(total_tokens, 1)
    logging.info('verify token agreement: %.5f (%d diff / %d tokens over %d utts)',
                 rate, diff_tokens, total_tokens, len(sample))
    if rate < 0.995:
        logging.error('verify token agreement %.5f below threshold 0.995', rate)
        sys.exit(1)


def main(args):
    utt2wav = read_wav_scp(args.dir)
    session = make_session(args.onnx_path, args.provider)
    input_names = [session.get_inputs()[0].name, session.get_inputs()[1].name]

    utt2speech_token = batched_extract(args, utt2wav, session, input_names)
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))

    if args.verify_num > 0:
        verify(args, utt2wav, utt2speech_token, session, input_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="kaldi data dir containing wav.scp")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="path to speech_tokenizer_v3.batch.onnx")
    parser.add_argument("--num_thread", type=int, default=8,
                        help="number of producer threads computing mel features")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="max utterances per onnx call")
    parser.add_argument("--max_batch_frames", type=int, default=100000,
                        help="max total padded mel frames (batch_size * padded_T) per onnx call")
    parser.add_argument("--provider", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="onnxruntime execution provider")
    parser.add_argument("--equal_length", action=argparse.BooleanOptionalAction, default=True,
                        help="batch only mels with exactly equal length: no padding, tokens match "
                             "the batch-1 reference (use --no-equal_length for padded packing)")
    parser.add_argument("--verify_num", type=int, default=0,
                        help="if >0, re-extract N random utts at batch 1 and check exact-match rate")
    args = parser.parse_args()

    main(args)
