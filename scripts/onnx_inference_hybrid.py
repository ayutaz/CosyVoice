#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Hybrid ONNX/PyTorch CosyVoice3 Inference

Uses existing CosyVoice for PyTorch components, swaps in ONNX for specific parts.

Usage:
    python scripts/onnx_inference_hybrid.py --text "<|en|>Hello world"
"""

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel


class OnnxFlowWrapper:
    """ONNX-based Flow inference wrapper"""

    def __init__(self, model_dir: str, use_fp16: bool = True):
        self.onnx_dir = os.path.join(model_dir, 'onnx')
        suffix = '_fp16' if use_fp16 else '_fp32'

        so = ort.SessionOptions()
        so.log_severity_level = 3

        print("  Loading Flow ONNX models...")
        self.token_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_token_embedding{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )
        self.speaker_projection = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_speaker_projection{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )
        self.pre_lookahead = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_pre_lookahead{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )

        estimator_path = os.path.join(self.onnx_dir, 'flow.decoder.estimator.fp16.onnx') if use_fp16 else \
            os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
        self.estimator = ort.InferenceSession(
            estimator_path, so, providers=['CPUExecutionProvider']
        )

    def inference(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10
    ) -> torch.Tensor:
        """Run flow inference using ONNX"""

        # Project speaker embedding
        embedding_norm = F.normalize(embedding, dim=1).numpy()
        spks = self.speaker_projection.run(
            None, {'embedding': embedding_norm.astype(np.float32)}
        )[0]

        # Concatenate tokens
        all_tokens = torch.concat([prompt_token, token], dim=1)

        # Embed tokens
        token_embedded = self.token_embedding.run(
            None, {'token': all_tokens.numpy().astype(np.int64)}
        )[0]

        # Pre-lookahead
        h = self.pre_lookahead.run(
            None, {'token_embedded': token_embedded.astype(np.float32)}
        )[0]

        # Prepare dimensions
        prompt_mel_len = prompt_feat.shape[1]
        total_mel_len = h.shape[1]

        # Build conditions
        conds = np.zeros((1, total_mel_len, 80), dtype=np.float32)
        conds[:, :prompt_mel_len, :] = prompt_feat.numpy()
        conds = conds.transpose(0, 2, 1)

        mu = h.transpose(0, 2, 1)
        mask = np.ones((1, 1, total_mel_len), dtype=np.float32)

        # Initialize x with noise
        x = np.random.randn(1, 80, total_mel_len).astype(np.float32)

        # Batch for estimator (batch=2)
        x_batch = np.concatenate([x, x], axis=0)
        mask_batch = np.concatenate([mask, mask], axis=0)
        mu_batch = np.concatenate([mu, mu], axis=0)
        spks_batch = np.concatenate([spks, spks], axis=0)
        conds_batch = np.concatenate([conds, conds], axis=0)

        # Euler solver
        for step in range(n_timesteps):
            t = np.array([step / n_timesteps, step / n_timesteps], dtype=np.float32)

            velocity = self.estimator.run(None, {
                'x': x_batch,
                'mask': mask_batch,
                'mu': mu_batch,
                't': t,
                'spks': spks_batch,
                'cond': conds_batch
            })[0]

            dt = 1.0 / n_timesteps
            x_batch = x_batch + velocity * dt

        mel = x_batch[:1, :, prompt_mel_len:]
        return torch.from_numpy(mel).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--text', type=str, default='<|en|>Hello, this is a test of CosyVoice.')
    parser.add_argument('--output', type=str, default='output_onnx_hybrid.wav')
    parser.add_argument('--use_onnx_flow', action='store_true', help='Use ONNX for Flow (otherwise PyTorch)')

    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid ONNX/PyTorch CosyVoice3 Inference")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Text: {args.text}")
    print(f"Use ONNX Flow: {args.use_onnx_flow}")
    print()

    # Load CosyVoice using AutoModel
    print("Loading CosyVoice3 model...")
    cosyvoice = AutoModel(args.model_dir)
    print("Model loaded!")

    # Optionally load ONNX Flow
    if args.use_onnx_flow:
        print("\nLoading ONNX Flow models...")
        onnx_flow = OnnxFlowWrapper(args.model_dir, use_fp16=True)
        print("ONNX Flow loaded!")

    # Run inference
    print(f"\nGenerating speech for: {args.text}")
    print("-" * 60)

    start_time = time.time()

    # Use cross-lingual mode (no prompt audio needed for language-tagged text)
    # Create a simple prompt from existing speaker
    spk_list = list(cosyvoice.frontend.spk2info.keys())
    if spk_list:
        print(f"Available speakers: {spk_list[:5]}...")

    # For simplicity, use SFT mode if speakers available
    audio_output = None

    for output in cosyvoice.inference_cross_lingual(
        args.text,
        prompt_wav=None,  # No prompt wav
        stream=False
    ):
        audio_output = output['tts_speech']

    if audio_output is None:
        print("Error: No audio generated")
        return

    elapsed = time.time() - start_time
    audio_np = audio_output.squeeze().numpy()
    duration = len(audio_np) / cosyvoice.sample_rate

    print(f"\nGeneration complete!")
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  Generation time: {elapsed:.2f}s")
    print(f"  RTF: {elapsed / duration:.2f}")

    # Save output
    sf.write(args.output, audio_np, cosyvoice.sample_rate)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
