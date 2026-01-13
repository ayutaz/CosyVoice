#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Working Hybrid ONNX/PyTorch CosyVoice3 Inference

Uses PyTorch for:
- Text embedding (Qwen2) - required for text processing
- HiFT vocoder - complete audio synthesis

Uses ONNX for:
- LLM backbone and decoder - speech token generation
- Flow matching - mel spectrogram generation

Usage:
    python scripts/onnx_inference_working.py --text "<|en|>Hello world"
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

from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd


class OnnxLLMWrapper:
    """ONNX-based LLM inference with PyTorch text embedding"""

    def __init__(self, model_dir: str, llm_pt: torch.nn.Module, use_fp16: bool = True):
        self.onnx_dir = os.path.join(model_dir, 'onnx')
        suffix = '_fp16' if use_fp16 else '_fp32'

        # Keep PyTorch LLM for text embedding
        self.llm_pt = llm_pt

        # ONNX session options
        so = ort.SessionOptions()
        so.log_severity_level = 3

        print("  Loading LLM ONNX models...")
        self.backbone_initial = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_backbone_initial{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )
        self.backbone_decode = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_backbone_decode{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )
        self.llm_decoder = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_decoder{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )
        self.speech_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_speech_embedding{suffix}.onnx'),
            so, providers=['CPUExecutionProvider']
        )

        # Model parameters for CosyVoice3
        self.hidden_dim = 896
        self.speech_token_size = 6561
        # CosyVoice3 uses speech_token_size + offset for special tokens
        self.sos = self.speech_token_size + 0      # 6561
        self.eos_token = self.speech_token_size + 1  # 6562
        self.task_id = self.speech_token_size + 2    # 6563

    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling_k: int = 25,
        max_len: int = 500,
        min_len: int = 10
    ):
        """Generate speech tokens using ONNX with PyTorch text embedding"""

        # Use PyTorch for text encoding (Qwen2) - CosyVoice3 style
        with torch.no_grad():
            # Concatenate prompt and text
            full_text = torch.concat([prompt_text, text], dim=1)
            full_text_len = text_len + prompt_text_len

            # Text embedding using PyTorch Qwen2 embed_tokens
            # CosyVoice3 directly uses embed_tokens without additional encoding
            text_emb = self.llm_pt.llm.model.model.embed_tokens(full_text)

            # Get SOS and task_id embeddings using speech_embedding
            # CosyVoice3 uses speech_embedding for special tokens (SOS, EOS, TASK_ID)
            sos_emb = self.llm_pt.speech_embedding.weight[self.sos].reshape(1, 1, -1)
            task_id_emb = self.llm_pt.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

            # Get prompt speech embedding
            if prompt_speech_token_len.item() > 0:
                prompt_speech_emb = self.llm_pt.speech_embedding(prompt_speech_token)
            else:
                prompt_speech_emb = torch.zeros(1, 0, self.hidden_dim, dtype=text_emb.dtype)

        # Build initial input: [SOS, text_emb, TASK_ID, prompt_speech_emb]
        # Note: CosyVoice3 doesn't use speaker embedding in LLM input
        lm_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1)
        lm_input_np = lm_input.detach().numpy().astype(np.float32)

        # Run initial ONNX forward pass
        seq_len = lm_input.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.float32)

        hidden_states = self.backbone_initial.run(
            None, {'inputs_embeds': lm_input_np, 'attention_mask': attention_mask}
        )[0]

        # Get initial prediction
        logits = self.llm_decoder.run(
            None, {'hidden_state': hidden_states[:, -1:, :]}
        )[0]

        # Calculate min/max length based on text
        text_only_len = text_len.item()
        min_len = max(min_len, int(text_only_len * 2))
        max_len = min(max_len, int(text_only_len * 20))

        print(f"    Text length: {text_only_len}, generating {min_len}-{max_len} tokens...")

        # Prepare KV cache (48 = 24 layers * 2 for key/value)
        # Initial hidden states need to be used to build cache
        past_key_values = np.zeros((48, 1, 2, seq_len, 64), dtype=np.float32)

        out_tokens = []
        for i in range(max_len):
            # Sample from logits
            logp = torch.from_numpy(logits).log_softmax(dim=-1).squeeze()
            top_k_logp, top_k_idx = torch.topk(logp, sampling_k)
            top_k_probs = torch.softmax(top_k_logp, dim=-1)
            sampled_idx = torch.multinomial(top_k_probs, 1)
            token_id = top_k_idx[sampled_idx].item()

            # Check EOS
            if token_id == self.eos_token and i >= min_len:
                break

            out_tokens.append(token_id)

            if i % 50 == 0:
                print(f"    Generated {i+1} tokens...")

            # Get next token embedding using ONNX
            next_emb = self.speech_embedding.run(
                None, {'token': np.array([[token_id]], dtype=np.int64)}
            )[0].astype(np.float32)

            # Update attention mask
            total_len = seq_len + len(out_tokens)
            attention_mask = np.ones((1, total_len), dtype=np.float32)

            # Decode step with KV cache
            outputs = self.backbone_decode.run(
                None, {
                    'inputs_embeds': next_emb,
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
        return torch.tensor([out_tokens], dtype=torch.int64)


class OnnxFlowWrapper:
    """ONNX-based Flow inference"""

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

        # Flow estimator
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
        """Convert speech tokens to mel spectrogram using ONNX Flow"""

        # Normalize and project speaker embedding
        embedding_norm = F.normalize(embedding, dim=1).numpy()
        spks = self.speaker_projection.run(
            None, {'embedding': embedding_norm.astype(np.float32)}
        )[0]

        # Concatenate prompt and generated tokens
        all_tokens = torch.concat([prompt_token, token], dim=1)

        # Embed tokens
        token_embedded = self.token_embedding.run(
            None, {'token': all_tokens.numpy().astype(np.int64)}
        )[0]

        # Pre-lookahead
        h = self.pre_lookahead.run(
            None, {'token_embedded': token_embedded.astype(np.float32)}
        )[0]

        # Dimensions
        prompt_mel_len = prompt_feat.shape[1]
        total_mel_len = h.shape[1]

        # Build conditions
        conds = np.zeros((1, total_mel_len, 80), dtype=np.float32)
        if prompt_mel_len > 0:
            conds[:, :prompt_mel_len, :] = prompt_feat.numpy()
        conds = conds.transpose(0, 2, 1)  # [batch, 80, mel_len]

        # Prepare mu and mask
        mu = h.transpose(0, 2, 1)  # [batch, 80, mel_len]
        mask = np.ones((1, 1, total_mel_len), dtype=np.float32)

        # Initialize x with noise
        x = np.random.randn(1, 80, total_mel_len).astype(np.float32)

        # Batch for estimator (needs batch=2)
        x_batch = np.concatenate([x, x], axis=0)
        mask_batch = np.concatenate([mask, mask], axis=0)
        mu_batch = np.concatenate([mu, mu], axis=0)
        spks_batch = np.concatenate([spks, spks], axis=0)
        conds_batch = np.concatenate([conds, conds], axis=0)

        print(f"    Running {n_timesteps} flow steps (mel_len={total_mel_len})...")

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

        # Return mel without prompt part
        mel = x_batch[:1, :, prompt_mel_len:]
        return torch.from_numpy(mel).float()


class HybridOnnxCosyVoice3:
    """Hybrid ONNX/PyTorch CosyVoice3 inference"""

    def __init__(self, model_dir: str, use_fp16: bool = True):
        self.model_dir = model_dir
        self.use_fp16 = use_fp16
        self.sample_rate = 24000
        self.device = 'cpu'

        # Load config
        yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
            self.configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path})

        self._load_pytorch_components()
        self._load_onnx_components()

    def _load_pytorch_components(self):
        """Load PyTorch components for text embedding and vocoder"""
        print("Loading PyTorch components...")

        # Load LLM for text embedding
        llm_config = self.configs['llm']
        llm_path = os.path.join(self.model_dir, 'llm.pt')
        print(f"  Loading LLM from {llm_path}...")
        state_dict = torch.load(llm_path, map_location=self.device)
        llm_config.load_state_dict(state_dict)
        llm_config.to(self.device)  # Move to CPU
        llm_config.eval()
        self.llm_pt = llm_config

        # Load HiFT vocoder
        hift_config = self.configs['hift']
        hift_path = os.path.join(self.model_dir, 'hift.pt')
        print(f"  Loading HiFT from {hift_path}...")
        hift_state_dict = {k.replace('generator.', ''): v
                          for k, v in torch.load(hift_path, map_location=self.device).items()}
        hift_config.load_state_dict(hift_state_dict)
        hift_config.to(self.device)  # Move to CPU
        hift_config.eval()
        self.hift = hift_config

        # Load frontend
        print("  Loading Frontend...")
        self.frontend = CosyVoiceFrontEnd(
            self.configs['get_tokenizer'],
            self.configs['feat_extractor'],
            os.path.join(self.model_dir, 'campplus.onnx'),
            os.path.join(self.model_dir, 'speech_tokenizer_v3.onnx'),
            os.path.join(self.model_dir, 'spk2info.pt'),
            self.configs['allowed_special']
        )

        print("  PyTorch components loaded!")

    def _load_onnx_components(self):
        """Load ONNX components for LLM and Flow"""
        print("\nLoading ONNX components...")
        self.onnx_llm = OnnxLLMWrapper(self.model_dir, self.llm_pt, self.use_fp16)
        self.onnx_flow = OnnxFlowWrapper(self.model_dir, self.use_fp16)
        print("  ONNX components loaded!")

    def inference_sft(self, text: str, spk_id: str, stream: bool = False):
        """SFT inference using pre-trained speaker"""
        print(f"\nInput text: {text}")
        print(f"Speaker: {spk_id}")

        # Normalize text
        normalized_text = self.frontend.text_normalize(text, split=False, text_frontend=True)
        print(f"Normalized: {normalized_text}")

        # Get model input from frontend
        model_input = self.frontend.frontend_sft(normalized_text, spk_id)

        return self._run_inference(model_input)

    def inference_cross_lingual(self, text: str, prompt_wav_path: str = None):
        """Cross-lingual inference (language tag in text)"""
        print(f"\nInput text: {text}")

        # Normalize text
        normalized_text = self.frontend.text_normalize(text, split=False, text_frontend=True)
        print(f"Normalized: {normalized_text}")

        # Check available speakers
        spk_list = list(self.frontend.spk2info.keys())
        if spk_list:
            print(f"Available speakers: {spk_list}")

        # Load prompt audio if provided
        if prompt_wav_path and os.path.exists(prompt_wav_path):
            import torchaudio
            prompt_wav, sr = torchaudio.load(prompt_wav_path)
            if sr != self.sample_rate:
                prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.sample_rate)
            prompt_wav = prompt_wav.mean(dim=0)  # mono

            # Get model input
            model_input = self.frontend.frontend_cross_lingual(
                normalized_text, prompt_wav, self.sample_rate, ''
            )
        elif spk_list:
            # Use first available speaker for SFT mode
            spk_id = spk_list[0]
            print(f"Using SFT mode with speaker: {spk_id}")
            model_input = self.frontend.frontend_sft(normalized_text, spk_id)
        else:
            # Create dummy input with random embedding
            print("No speakers available, using random embedding")
            text_token, text_token_len = self.frontend._extract_text_token(normalized_text)
            model_input = {
                'text': text_token,
                'text_len': text_token_len,
                'prompt_text': torch.zeros(1, 0, dtype=torch.int32),
                'prompt_text_len': torch.tensor([0], dtype=torch.int32),
                'llm_prompt_speech_token': torch.zeros(1, 0, dtype=torch.int32),
                'llm_prompt_speech_token_len': torch.tensor([0], dtype=torch.int32),
                'flow_prompt_speech_token': torch.zeros(1, 0, dtype=torch.int32),
                'flow_prompt_speech_token_len': torch.tensor([0], dtype=torch.int32),
                'prompt_speech_feat': torch.zeros(1, 0, 80),
                'prompt_speech_feat_len': torch.tensor([0], dtype=torch.int32),
                'llm_embedding': torch.randn(1, 192),
                'flow_embedding': torch.randn(1, 192),
            }

        return self._run_inference(model_input)

    def _run_inference(self, model_input: dict):
        """Run full TTS pipeline using ONNX LLM and Flow"""
        start_time = time.time()

        # Extract inputs and ensure they are on CPU
        text = model_input['text'].cpu()
        text_len = torch.tensor([text.shape[1]], dtype=torch.int32)
        prompt_text = model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)).cpu()
        prompt_text_len = torch.tensor([prompt_text.shape[1]], dtype=torch.int32)
        llm_prompt_speech_token = model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)).cpu()
        llm_prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32)
        flow_prompt_speech_token = model_input.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)).cpu()
        prompt_speech_feat = model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80)).cpu()
        llm_embedding = model_input.get('llm_embedding', torch.zeros(1, 192)).cpu()
        flow_embedding = model_input.get('flow_embedding', torch.zeros(1, 192)).cpu()

        print(f"\n  Text tokens: {text.shape[1]}")
        print(f"  Prompt speech tokens: {llm_prompt_speech_token.shape[1]}")
        print(f"  Prompt mel frames: {prompt_speech_feat.shape[1]}")

        # Step 1: LLM inference (ONNX with PyTorch text embedding)
        print("\n1. Running LLM (ONNX)...")
        llm_start = time.time()
        speech_tokens = self.onnx_llm.inference(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=llm_prompt_speech_token,
            prompt_speech_token_len=llm_prompt_speech_token_len,
            embedding=llm_embedding,
            sampling_k=25,
            max_len=500,
            min_len=10
        )
        llm_time = time.time() - llm_start
        print(f"   LLM time: {llm_time:.2f}s")

        # Step 2: Flow inference (ONNX)
        print("\n2. Running Flow (ONNX)...")
        flow_start = time.time()
        mel = self.onnx_flow.inference(
            token=speech_tokens,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            n_timesteps=10
        )
        flow_time = time.time() - flow_start
        print(f"   Mel shape: {mel.shape}")
        print(f"   Flow time: {flow_time:.2f}s")

        # Step 3: HiFT vocoder (PyTorch)
        print("\n3. Running HiFT (PyTorch)...")
        hift_start = time.time()
        with torch.no_grad():
            # CosyVoice3 uses CausalHiFTGenerator with finalize parameter
            audio, _ = self.hift.inference(speech_feat=mel, finalize=True)
        hift_time = time.time() - hift_start
        print(f"   Audio shape: {audio.shape}")
        print(f"   HiFT time: {hift_time:.2f}s")

        total_time = time.time() - start_time
        audio_len = audio.shape[1] / self.sample_rate

        print(f"\n" + "=" * 50)
        print(f"Total time: {total_time:.2f}s")
        print(f"Audio length: {audio_len:.2f}s")
        print(f"RTF: {total_time / audio_len:.2f}")
        print("=" * 50)

        return audio.squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--text', type=str, default='<|en|>Hello, this is a test of the CosyVoice system.')
    parser.add_argument('--output', type=str, default='output_onnx_working.wav')
    parser.add_argument('--prompt_wav', type=str, default=None, help='Optional prompt audio')
    parser.add_argument('--spk', type=str, default=None, help='Speaker ID for SFT mode')
    parser.add_argument('--fp32', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid ONNX/PyTorch CosyVoice3 Inference")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Text: {args.text}")
    print(f"Use FP16: {not args.fp32}")
    print()

    # Create engine
    engine = HybridOnnxCosyVoice3(args.model_dir, use_fp16=not args.fp32)

    # Run inference
    if args.spk:
        # SFT mode with pre-trained speaker
        audio = engine.inference_sft(args.text, args.spk)
    else:
        # Cross-lingual mode
        audio = engine.inference_cross_lingual(args.text, args.prompt_wav)

    # Save output
    sf.write(args.output, audio.numpy(), engine.sample_rate)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
