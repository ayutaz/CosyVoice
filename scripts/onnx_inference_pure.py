#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Pure ONNX CosyVoice3 Inference (No PyTorch model loading)

This script performs TTS inference using ONLY ONNX models.
Suitable for porting to Unity Sentis.

Usage:
    python scripts/onnx_inference_pure.py --text "<|en|>Hello world"
"""

import argparse
import os
import sys
from pathlib import Path
import time
import json

import numpy as np
import onnxruntime as ort
import soundfile as sf

# Add project root to path (only for tokenizer)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PureOnnxCosyVoice3:
    """Pure ONNX-based CosyVoice3 inference (no PyTorch)"""

    def __init__(self, model_dir: str, use_fp16: bool = True):
        self.model_dir = model_dir
        self.onnx_dir = os.path.join(model_dir, 'onnx')
        self.use_fp16 = use_fp16
        self.sample_rate = 24000

        # Model parameters for CosyVoice3
        self.hidden_dim = 896
        self.speech_token_size = 6561
        self.sos = self.speech_token_size + 0      # 6561
        self.eos_token = self.speech_token_size + 1  # 6562
        self.task_id = self.speech_token_size + 2    # 6563

        self._load_tokenizer()
        self._load_onnx_models()

    def _load_tokenizer(self):
        """Load Qwen2 tokenizer for text processing"""
        print("Loading tokenizer...")
        from transformers import AutoTokenizer

        qwen_path = os.path.join(self.model_dir, 'CosyVoice-BlankEN')
        if os.path.exists(qwen_path):
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
        print("  Tokenizer loaded!")

    def _load_onnx_models(self):
        """Load all ONNX models"""
        suffix = '_fp16' if self.use_fp16 else '_fp32'

        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = ['CPUExecutionProvider']

        print("Loading ONNX models...")

        # Text embedding
        self.text_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'text_embedding_fp32.onnx'),
            so, providers=providers
        )
        print("  Text embedding loaded")

        # LLM models
        self.llm_backbone_initial = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_backbone_initial{suffix}.onnx'),
            so, providers=providers
        )
        self.llm_backbone_decode = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_backbone_decode{suffix}.onnx'),
            so, providers=providers
        )
        self.llm_decoder = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_decoder{suffix}.onnx'),
            so, providers=providers
        )
        self.llm_speech_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_speech_embedding{suffix}.onnx'),
            so, providers=providers
        )
        print("  LLM models loaded")

        # Flow models
        self.flow_token_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_token_embedding{suffix}.onnx'),
            so, providers=providers
        )
        self.flow_speaker_projection = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_speaker_projection{suffix}.onnx'),
            so, providers=providers
        )
        self.flow_pre_lookahead = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_pre_lookahead{suffix}.onnx'),
            so, providers=providers
        )
        self.flow_estimator = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'flow.decoder.estimator.fp16.onnx') if self.use_fp16 else
            os.path.join(self.model_dir, 'flow.decoder.estimator.fp32.onnx'),
            so, providers=providers
        )
        print("  Flow models loaded")

        # HiFT models (always use FP32 for numerical stability)
        # F0 predictor has large intermediate values that cause FP16 overflow
        self.hift_f0_predictor = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'hift_f0_predictor_fp32.onnx'),
            so, providers=providers
        )
        self.hift_source_generator = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'hift_source_generator_fp32.onnx'),
            so, providers=providers
        )
        self.hift_decoder = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'hift_decoder_fp32.onnx'),
            so, providers=providers
        )
        print("  HiFT models loaded")

        print("All ONNX models loaded!")

    def tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text using Qwen2 tokenizer"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return np.array([tokens], dtype=np.int64)

    def get_text_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings using ONNX"""
        return self.text_embedding.run(None, {'input_ids': token_ids})[0]

    def get_speech_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get speech token embeddings using ONNX"""
        return self.llm_speech_embedding.run(None, {'token': token_ids})[0]

    def llm_inference(
        self,
        text: str,
        sampling_k: int = 25,
        max_len: int = 500,
        min_len: int = 10
    ) -> np.ndarray:
        """Generate speech tokens using pure ONNX"""
        print("  Tokenizing text...")

        # Tokenize text
        text_tokens = self.tokenize_text(text)
        text_len = text_tokens.shape[1]
        print(f"    Text tokens: {text_len}")

        # Get text embedding
        text_emb = self.get_text_embedding(text_tokens)

        # Get SOS and TASK_ID embeddings
        sos_emb = self.get_speech_embedding(np.array([[self.sos]], dtype=np.int64))
        task_id_emb = self.get_speech_embedding(np.array([[self.task_id]], dtype=np.int64))

        # Build initial input: [SOS, text_emb, TASK_ID]
        lm_input = np.concatenate([sos_emb, text_emb, task_id_emb], axis=1).astype(np.float32)

        # Initial forward pass
        seq_len = lm_input.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.float32)

        initial_outputs = self.llm_backbone_initial.run(
            None, {'inputs_embeds': lm_input, 'attention_mask': attention_mask}
        )
        hidden_states = initial_outputs[0]

        # Get KV cache from initial pass (if available)
        if len(initial_outputs) > 1:
            past_key_values = initial_outputs[1]
            print(f"    Initial KV cache shape: {past_key_values.shape}")
        else:
            # Fallback: create zeros cache (old behavior, will cause issues)
            print("    WARNING: No KV cache from initial pass, using zeros")
            past_key_values = np.zeros((48, 1, 2, seq_len, 64), dtype=np.float32)

        # Get initial logits
        logits = self.llm_decoder.run(
            None, {'hidden_state': hidden_states[:, -1:, :]}
        )[0]

        # Calculate min/max length
        min_len = max(min_len, int(text_len * 2))
        max_len = min(max_len, int(text_len * 20))
        print(f"    Generating {min_len}-{max_len} tokens...")

        out_tokens = []
        for i in range(max_len):
            # Sample from logits (top-k sampling)
            logp = self._log_softmax(logits.squeeze())
            top_k_idx = np.argsort(logp)[-sampling_k:]
            top_k_logp = logp[top_k_idx]
            top_k_probs = self._softmax(top_k_logp)
            sampled_idx = np.random.choice(len(top_k_idx), p=top_k_probs)
            token_id = top_k_idx[sampled_idx]

            # Check EOS
            if token_id == self.eos_token and i >= min_len:
                break

            out_tokens.append(token_id)

            if i % 50 == 0:
                print(f"    Generated {i+1} tokens...")

            # Get next token embedding
            next_emb = self.get_speech_embedding(np.array([[token_id]], dtype=np.int64))

            # Update attention mask
            total_len = seq_len + len(out_tokens)
            attention_mask = np.ones((1, total_len), dtype=np.float32)

            # Decode step
            outputs = self.llm_backbone_decode.run(
                None, {
                    'inputs_embeds': next_emb.astype(np.float32),
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values
                }
            )
            hidden_states = outputs[0]
            past_key_values = outputs[1]

            # Get next logits
            logits = self.llm_decoder.run(
                None, {'hidden_state': hidden_states}
            )[0]

        print(f"    Generated {len(out_tokens)} speech tokens")
        return np.array([out_tokens], dtype=np.int64)

    def flow_inference(
        self,
        speech_tokens: np.ndarray,
        embedding: np.ndarray,
        n_timesteps: int = 10
    ) -> np.ndarray:
        """Convert speech tokens to mel using pure ONNX Flow"""

        # Normalize and project speaker embedding
        embedding_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        spks = self.flow_speaker_projection.run(
            None, {'embedding': embedding_norm.astype(np.float32)}
        )[0]

        # Embed tokens
        token_embedded = self.flow_token_embedding.run(
            None, {'token': speech_tokens.astype(np.int64)}
        )[0]

        # Pre-lookahead
        h = self.flow_pre_lookahead.run(
            None, {'token_embedded': token_embedded.astype(np.float32)}
        )[0]

        # Dimensions
        mel_len = h.shape[1]

        # Build conditions (no prompt)
        conds = np.zeros((1, 80, mel_len), dtype=np.float32)

        # Prepare mu and mask
        mu = h.transpose(0, 2, 1)
        mask = np.ones((1, 1, mel_len), dtype=np.float32)

        # Initialize x with noise
        x = np.random.randn(1, 80, mel_len).astype(np.float32)

        # Batch for estimator (needs batch=2)
        x_batch = np.concatenate([x, x], axis=0)
        mask_batch = np.concatenate([mask, mask], axis=0)
        mu_batch = np.concatenate([mu, mu], axis=0)
        spks_batch = np.concatenate([spks, spks], axis=0)
        conds_batch = np.concatenate([conds, conds], axis=0)

        print(f"    Running {n_timesteps} flow steps (mel_len={mel_len})...")

        # Euler solver
        for step in range(n_timesteps):
            t = np.array([step / n_timesteps, step / n_timesteps], dtype=np.float32)

            velocity = self.flow_estimator.run(None, {
                'x': x_batch,
                'mask': mask_batch,
                'mu': mu_batch,
                't': t,
                'spks': spks_batch,
                'cond': conds_batch
            })[0]

            dt = 1.0 / n_timesteps
            x_batch = x_batch + velocity * dt

        mel = x_batch[:1]
        return mel

    def _stft(self, x: np.ndarray, n_fft: int = 16, hop_len: int = 4) -> tuple:
        """Compute STFT using torch (for exact PyTorch HiFT compatibility)"""
        import torch
        from scipy.signal import get_window

        # Create Hann window matching PyTorch HiFT
        window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))

        # Convert signal to tensor
        x_t = torch.from_numpy(x.astype(np.float32))
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        # Compute STFT (same as HiFT._stft)
        spec = torch.stft(x_t, n_fft, hop_len, n_fft, window=window, return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, T, 2]

        # Return real and imaginary parts
        real = spec[..., 0].numpy()
        imag = spec[..., 1].numpy()

        return real.squeeze(0), imag.squeeze(0)

    def _istft(self, magnitude: np.ndarray, phase: np.ndarray, n_fft: int = 16, hop_len: int = 4) -> np.ndarray:
        """Compute ISTFT from magnitude and phase using torch (for exact PyTorch compatibility)"""
        import torch
        from scipy.signal import get_window

        # Create Hann window matching PyTorch HiFT
        window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))

        # Clip magnitude to prevent numerical issues (same as HiFT._istft)
        magnitude = np.clip(magnitude, a_min=None, a_max=100.0)

        # Reconstruct complex STFT from magnitude and phase
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)

        # Convert to torch tensors
        real_t = torch.from_numpy(real.astype(np.float32))
        imag_t = torch.from_numpy(imag.astype(np.float32))
        complex_spec = torch.complex(real_t, imag_t)

        # Compute ISTFT (same as HiFT._istft)
        audio = torch.istft(complex_spec, n_fft, hop_len, n_fft, window=window)

        return audio.numpy()

    def hift_inference(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel to audio using pure ONNX HiFT"""

        # Predict F0
        f0 = self.hift_f0_predictor.run(None, {'mel': mel.astype(np.float32)})[0]
        print(f"    F0 shape: {f0.shape}, range: [{f0.min():.1f}, {f0.max():.1f}]")

        # Generate source from F0
        # F0 needs to be reshaped to [batch, 1, time]
        f0_input = f0[:, np.newaxis, :]  # [batch, 1, time]
        source = self.hift_source_generator.run(None, {'f0': f0_input.astype(np.float32)})[0]
        print(f"    Source shape: {source.shape}")

        # Compute STFT of source signal
        # HiFT uses n_fft=16, hop_len=4
        source_squeezed = source.squeeze()  # [time]
        try:
            stft_real, stft_imag = self._stft(source_squeezed, n_fft=16, hop_len=4)
            # Combine real and imaginary: [batch, 18, time] (9 real + 9 imaginary)
            source_stft = np.concatenate([stft_real, stft_imag], axis=0)
            source_stft = source_stft[np.newaxis, :, :]  # Add batch dim
            print(f"    Source STFT shape: {source_stft.shape}")

            # Decode mel + source_stft to magnitude and phase
            outputs = self.hift_decoder.run(None, {
                'mel': mel.astype(np.float32),
                'source_stft': source_stft.astype(np.float32)
            })
            magnitude = outputs[0]  # [batch, 9, time]
            phase = outputs[1]      # [batch, 9, time]
            print(f"    Magnitude shape: {magnitude.shape}, Phase shape: {phase.shape}")

            # Apply ISTFT to convert magnitude and phase to audio
            audio = self._istft(magnitude.squeeze(0), phase.squeeze(0), n_fft=16, hop_len=4)

            # Apply audio clipping (same as HiFT: audio_limit=0.99)
            audio = np.clip(audio, -0.99, 0.99)

            print(f"    Audio shape: {audio.shape}")
            return audio
        except Exception as e:
            print(f"    HiFT decoder failed: {e}")
            print("    Using Griffin-Lim fallback...")
            return self._griffin_lim(mel)

    def _griffin_lim(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Simple Griffin-Lim vocoder as fallback"""
        try:
            import librosa
            mel_db = mel.squeeze()
            # Assume mel is in linear scale, convert to power
            mel_power = np.maximum(mel_db, 1e-10)
            audio = librosa.feature.inverse.mel_to_audio(
                mel_power,
                sr=self.sample_rate,
                hop_length=256,
                n_fft=1024,
                n_iter=n_iter
            )
            return audio
        except Exception as e:
            print(f"    Griffin-Lim failed: {e}")
            # Return noise as last resort
            audio_len = mel.shape[2] * 256
            return np.random.randn(audio_len).astype(np.float32) * 0.01

    def inference(self, text: str) -> np.ndarray:
        """Full TTS inference using pure ONNX"""
        print(f"\nInput text: {text}")
        start_time = time.time()

        # Step 1: LLM inference
        print("\n1. Running LLM (ONNX)...")
        llm_start = time.time()
        speech_tokens = self.llm_inference(text, sampling_k=25, max_len=500, min_len=10)
        llm_time = time.time() - llm_start
        print(f"   LLM time: {llm_time:.2f}s")

        # Step 2: Flow inference
        print("\n2. Running Flow (ONNX)...")
        flow_start = time.time()
        # Use random speaker embedding (no prompt)
        embedding = np.random.randn(1, 192).astype(np.float32)
        mel = self.flow_inference(speech_tokens, embedding, n_timesteps=10)
        flow_time = time.time() - flow_start
        print(f"   Mel shape: {mel.shape}")
        print(f"   Flow time: {flow_time:.2f}s")

        # Step 3: HiFT vocoder
        print("\n3. Running HiFT (ONNX)...")
        hift_start = time.time()
        audio = self.hift_inference(mel)
        hift_time = time.time() - hift_start
        print(f"   HiFT time: {hift_time:.2f}s")

        total_time = time.time() - start_time
        audio_len = len(audio) / self.sample_rate

        print(f"\n" + "=" * 50)
        print(f"Total time: {total_time:.2f}s")
        print(f"Audio length: {audio_len:.2f}s")
        print(f"RTF: {total_time / audio_len:.2f}")
        print("=" * 50)

        return audio

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def _log_softmax(x):
        return x - np.log(np.sum(np.exp(x - np.max(x)))) - np.max(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--text', type=str, default='<|en|>Hello, this is a test of pure ONNX inference.')
    parser.add_argument('--output', type=str, default='output_onnx_pure.wav')
    parser.add_argument('--fp32', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("Pure ONNX CosyVoice3 Inference")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Text: {args.text}")
    print(f"Use FP16: {not args.fp32}")
    print()

    # Create engine
    engine = PureOnnxCosyVoice3(args.model_dir, use_fp16=not args.fp32)

    # Run inference
    audio = engine.inference(args.text)

    # Save output
    sf.write(args.output, audio, engine.sample_rate)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
