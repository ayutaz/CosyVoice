#!/usr/bin/env python3
"""Prepare LibriTTS data for CosyVoice2 draft model training.

Downloads LibriTTS subsets and converts to CosyVoice parquet format
compatible with the training pipeline (parquet_opener -> tokenize -> filter -> ...).

Required parquet columns:
  - utt (str): utterance ID
  - text (str): normalized text
  - audio_data (bytes): raw wav bytes (read by filter via torchaudio.load(BytesIO(...)))
  - speech_token (list[int]): acoustic tokens from speech_tokenizer ONNX
  - utt_embedding (list[float]): speaker embedding from campplus ONNX

Usage:
    # Docker GPU (recommended):
    docker run --rm --gpus all \\
        -v cosyvoice-data:/data \\
        -v ./data/libritts:/output \\
        cosyvoice-gpu python3.10 scripts/prepare_libritts.py \\
        --subsets train-clean-100 \\
        --output_dir /output \\
        --model_dir /data/models \\
        --libritts_root /data
"""

import argparse
import io
import os
import time

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LibriTTS for CosyVoice2")
    parser.add_argument("--subsets", default="test-clean",
                        help="Comma-separated LibriTTS subsets")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_dir", required=True,
                        help="CosyVoice2 model directory (for campplus/speech_tokenizer ONNX)")
    parser.add_argument("--libritts_root", default=None,
                        help="Existing LibriTTS root (skip download if provided)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per subset (-1 for all)")
    parser.add_argument("--eval_list_output", default=None,
                        help="If set, output eval list file (utt_id\\ttext\\tprompt_wav)")
    parser.add_argument("--sample_rate", type=int, default=24000,
                        help="Target sample rate for audio_data")
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Write parquet in chunks of this size (saves memory)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for parallel processing (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel processing")
    parser.add_argument("--onnx_threads", type=int, default=4,
                        help="Number of ONNX intra-op threads")
    return parser.parse_args()


def download_libritts(subset, download_dir):
    """Download LibriTTS subset using torchaudio."""
    print("Downloading LibriTTS subset: {} ...".format(subset))
    os.makedirs(download_dir, exist_ok=True)
    dataset = torchaudio.datasets.LIBRITTS(root=download_dir, url=subset, download=True)
    print("  Downloaded {} utterances".format(len(dataset)))
    return dataset


def find_libritts_files(libritts_root, subset):
    """Find all .wav files in a LibriTTS subset directory."""
    subset_dir = os.path.join(libritts_root, "LibriTTS", subset)
    if not os.path.exists(subset_dir):
        subset_dir = os.path.join(libritts_root, subset)
    if not os.path.exists(subset_dir):
        raise FileNotFoundError("Cannot find subset dir: {}".format(subset_dir))

    print("  Scanning directory: {} ...".format(subset_dir))
    t0 = time.time()
    samples = []
    for root, dirs, files in os.walk(subset_dir):
        for f in sorted(files):
            if f.endswith(".normalized.txt"):
                wav_name = f.replace(".normalized.txt", ".wav")
                wav_path = os.path.join(root, wav_name)
                txt_path = os.path.join(root, f)
                if os.path.exists(wav_path):
                    with open(txt_path, "r", encoding="utf-8") as fh:
                        text = fh.read().strip()
                    utt_id = f.replace(".normalized.txt", "")
                    samples.append({
                        "utt_id": utt_id,
                        "wav_path": wav_path,
                        "text": text,
                    })
    print("  Found {} files in {:.1f}s".format(len(samples), time.time() - t0))
    return samples


def create_eval_list(samples, output_path, max_samples=-1):
    """Create evaluation list file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if max_samples > 0:
        samples = samples[:max_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write("{}\t{}\t{}\n".format(s["utt_id"], s["text"], s["wav_path"]))

    print("Eval list saved: {} ({} samples)".format(output_path, len(samples)))


class LightweightFeatureExtractor:
    """Lightweight ONNX-only feature extractor for data preparation."""

    def __init__(self, model_dir, onnx_threads=4):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = onnx_threads

        campplus_path = os.path.join(model_dir, "campplus.onnx")
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

        tokenizer_path = os.path.join(model_dir, "speech_tokenizer_v2.onnx")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.join(model_dir, "speech_tokenizer.onnx")
        # Use GPU if available (requires fixed 3000-frame input for cuDNN JIT efficiency)
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            tokenizer_path, sess_options=option, providers=[provider]
        )

        actual_provider = self.speech_tokenizer_session.get_providers()[0]
        print("LightweightFeatureExtractor initialized (threads={}):".format(onnx_threads))
        print("  campplus: {} (CPU)".format(campplus_path))
        print("  speech_tokenizer: {} ({})".format(tokenizer_path, actual_provider))

        # Warmup GPU with fixed-size input to trigger single cuDNN JIT compilation
        if actual_provider == "CUDAExecutionProvider":
            print("  Warming up GPU (cuDNN JIT compile for 3000 frames)...")
            t0 = time.time()
            dummy_feat = np.zeros((1, 128, self.WHISPER_N_FRAMES), dtype=np.float32)
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0].name: dummy_feat,
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array([self.WHISPER_N_FRAMES], dtype=np.int32),
                },
            )
            print("  Warmup done in {:.1f}s".format(time.time() - t0))

        # Cache resampler instances
        self._resamplers = {}

    def _get_resampler(self, orig_freq, new_freq):
        key = (orig_freq, new_freq)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=orig_freq, new_freq=new_freq
            )
        return self._resamplers[key]

    # Whisper's 30-second constants
    WHISPER_N_SAMPLES = 480000  # 30s at 16kHz
    WHISPER_N_FRAMES = 3000    # 480000 / 160 hop

    def extract_speech_token(self, speech_16k):
        """Extract speech tokens from 16kHz audio tensor (1, T).

        Pads to 30s with feat_len=3000 to avoid onnxruntime CPU dynamic shape bug.
        """
        actual_token_len = speech_16k.shape[-1] // 160 // 4  # model downsamples 4x
        audio_padded = whisper.pad_or_trim(speech_16k.flatten(), self.WHISPER_N_SAMPLES)
        feat = whisper.log_mel_spectrogram(audio_padded.unsqueeze(0), n_mels=128)
        speech_token = self.speech_tokenizer_session.run(
            None,
            {
                self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                self.speech_tokenizer_session.get_inputs()[1].name: np.array([self.WHISPER_N_FRAMES], dtype=np.int32),
            },
        )[0].flatten().tolist()
        return speech_token[:actual_token_len]

    def extract_spk_embedding(self, speech_16k):
        """Extract speaker embedding from 16kHz audio tensor (1, T)."""
        feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(
            None,
            {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0].flatten().tolist()
        return embedding

    def process_sample(self, wav_path, sample_rate=24000):
        """Process a single wav file and return (speech_token, utt_embedding, audio_data).

        Reads the wav file only once, resamples to both 16kHz and target sample_rate.
        """
        speech, sr = torchaudio.load(wav_path)
        speech = speech.mean(dim=0, keepdim=True)

        # Resample to 16kHz for feature extraction
        if sr != 16000:
            speech_16k = self._get_resampler(sr, 16000)(speech)
        else:
            speech_16k = speech

        # Skip audio longer than 30s
        duration = speech_16k.shape[1] / 16000
        if duration > 30:
            return None, None, None, duration

        # Extract features
        speech_token = self.extract_speech_token(speech_16k)
        utt_embedding = self.extract_spk_embedding(speech_16k)

        # Resample to target sample_rate for audio_data
        if sr != sample_rate:
            speech_out = self._get_resampler(sr, sample_rate)(speech)
        else:
            speech_out = speech
        buf = io.BytesIO()
        torchaudio.save(buf, speech_out, sample_rate, format="wav")
        audio_data = buf.getvalue()

        return speech_token, utt_embedding, audio_data, duration


def process_to_parquet(samples, output_path, model_dir, sample_rate=24000,
                       max_samples=-1, chunk_size=5000, onnx_threads=4):
    """Convert samples to CosyVoice training parquet format."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if max_samples > 0:
        samples = samples[:max_samples]

    print("Initializing lightweight feature extractor...")
    extractor = LightweightFeatureExtractor(model_dir, onnx_threads=onnx_threads)

    total = len(samples)
    records = []
    skipped = 0
    writer = None
    schema = pa.schema([
        ("utt", pa.string()),
        ("text", pa.string()),
        ("audio_data", pa.binary()),
        ("speech_token", pa.list_(pa.int32())),
        ("utt_embedding", pa.list_(pa.float32())),
        ("spk_embedding", pa.list_(pa.float32())),
    ])

    t_start = time.time()
    t_last_report = t_start

    for idx, sample in enumerate(samples):
        try:
            speech_token, utt_embedding, audio_data, duration = extractor.process_sample(
                sample["wav_path"], sample_rate
            )

            if speech_token is None:
                print("  Skipping {} (duration {:.1f}s > 30s)".format(sample["utt_id"], duration))
                skipped += 1
                continue

            records.append({
                "utt": sample["utt_id"],
                "text": sample["text"],
                "audio_data": audio_data,
                "speech_token": speech_token,
                "utt_embedding": utt_embedding,
                "spk_embedding": utt_embedding,
            })
        except Exception as e:
            print("Warning: failed to process {}: {}".format(sample["utt_id"], e))
            skipped += 1
            continue

        # Write chunk to parquet incrementally
        if len(records) >= chunk_size:
            batch = _records_to_batch(records, schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(batch)
            print("  Flushed chunk ({} records written so far)".format(
                (idx + 1) - skipped))
            records = []

        # Progress report every 10 seconds
        now = time.time()
        if now - t_last_report >= 10:
            done = idx + 1
            elapsed = now - t_start
            speed = done / elapsed
            eta = (total - done) / speed if speed > 0 else 0
            print("  [{}/{}] {:.1f} samples/s, elapsed {:.0f}s, ETA {:.0f}s, skipped {}".format(
                done, total, speed, elapsed, eta, skipped))
            t_last_report = now

    # Write remaining records
    if records:
        batch = _records_to_batch(records, schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(batch)

    if writer is not None:
        writer.close()

    total_written = (len(samples) if max_samples < 0 else min(max_samples, len(samples))) - skipped
    elapsed = time.time() - t_start
    print("Parquet saved: {} ({} records, {:.0f}s, {:.1f} samples/s)".format(
        output_path, total_written, elapsed, total_written / elapsed if elapsed > 0 else 0))


def _records_to_batch(records, schema):
    """Convert list of record dicts to a pyarrow Table."""
    import pyarrow as pa
    return pa.table({
        "utt": pa.array([r["utt"] for r in records], type=pa.string()),
        "text": pa.array([r["text"] for r in records], type=pa.string()),
        "audio_data": pa.array([r["audio_data"] for r in records], type=pa.binary()),
        "speech_token": pa.array([r["speech_token"] for r in records], type=pa.list_(pa.int32())),
        "utt_embedding": pa.array([r["utt_embedding"] for r in records], type=pa.list_(pa.float32())),
        "spk_embedding": pa.array([r["spk_embedding"] for r in records], type=pa.list_(pa.float32())),
    })


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(",")]

    for subset in subsets:
        print("\n=== Processing subset: {} ===".format(subset))

        if args.libritts_root:
            samples = find_libritts_files(args.libritts_root, subset)
        else:
            download_dir = os.path.join(args.output_dir, "raw")
            dataset = download_libritts(subset, download_dir)
            samples = find_libritts_files(download_dir, subset)

        print("Found {} samples in {}".format(len(samples), subset))

        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        if args.eval_list_output:
            create_eval_list(samples, args.eval_list_output, args.max_samples)

        # Shard the samples for parallel processing
        if args.num_shards > 1:
            shard_samples = samples[args.shard_id::args.num_shards]
            print("Shard {}/{}: {} samples (of {} total)".format(
                args.shard_id, args.num_shards, len(shard_samples), len(samples)))
            suffix = "{}_shard{}.parquet".format(subset.replace("-", "_"), args.shard_id)
        else:
            shard_samples = samples
            suffix = "{}.parquet".format(subset.replace("-", "_"))

        parquet_path = os.path.join(args.output_dir, suffix)
        process_to_parquet(shard_samples, parquet_path, args.model_dir,
                           sample_rate=args.sample_rate, max_samples=args.max_samples,
                           chunk_size=args.chunk_size, onnx_threads=args.onnx_threads)


if __name__ == "__main__":
    main()
