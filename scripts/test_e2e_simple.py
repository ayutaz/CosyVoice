#!/usr/bin/env python3
"""
Simple End-to-End test: Compare PyTorch vs ONNX inference outputs
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'


def main():
    print("=" * 60)
    print("Simple End-to-End Comparison")
    print("=" * 60)

    # Load CosyVoice
    from cosyvoice.cli.cosyvoice import CosyVoice3
    cosyvoice = CosyVoice3(MODEL_DIR)

    test_text = "<|en|>Hello world."

    # Generate with PyTorch
    print("\n1. PyTorch Inference:")
    for i, output in enumerate(cosyvoice.inference_cross_lingual(
        test_text,
        prompt_wav=None,
        stream=False,
        text_frontend=False
    )):
        audio_pt = output['tts_speech']
        print(f"  Audio shape: {audio_pt.shape}")
        print(f"  Duration: {audio_pt.shape[1] / cosyvoice.sample_rate:.2f}s")
        sf.write('output_pytorch_test.wav', audio_pt.squeeze().numpy(), cosyvoice.sample_rate)
        print("  Saved: output_pytorch_test.wav")
        break

    # Now run pure ONNX with same text
    print("\n2. Pure ONNX Inference:")
    print("  (Run: python scripts/onnx_inference_pure.py --text '<|en|>Hello world.' --output output_onnx_test.wav)")

    # Check what the LLM is actually generating
    print("\n3. Checking LLM behavior:")
    with torch.no_grad():
        # Get text tokens
        from transformers import AutoTokenizer
        qwen_path = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

        text_tokens = tokenizer.encode(test_text, add_special_tokens=False)
        print(f"  Text tokens: {text_tokens}")
        print(f"  Text token count: {len(text_tokens)}")

        # Get embeddings
        text_token_tensor = torch.tensor([text_tokens], dtype=torch.long)
        text_emb = cosyvoice.model.llm.llm.model.model.embed_tokens(text_token_tensor)
        print(f"  Text embedding shape: {text_emb.shape}")

        # Special tokens
        speech_token_size = 6561
        print(f"\n  Speech token size: {speech_token_size}")
        print(f"  SOS token: {speech_token_size}")
        print(f"  EOS token: {speech_token_size + 1}")
        print(f"  TASK_ID token: {speech_token_size + 2}")

        # Check speech embedding weights
        print(f"\n  Speech embedding weight shape: {cosyvoice.model.llm.speech_embedding.weight.shape}")

    # Compare audio files if both exist
    if os.path.exists('output_pytorch_test.wav') and os.path.exists('output_onnx_test.wav'):
        print("\n4. Audio Comparison:")
        audio_pt, sr_pt = sf.read('output_pytorch_test.wav')
        audio_onnx, sr_onnx = sf.read('output_onnx_test.wav')

        print(f"  PyTorch: {len(audio_pt)/sr_pt:.2f}s")
        print(f"  ONNX: {len(audio_onnx)/sr_onnx:.2f}s")
        print(f"  Duration ratio: {len(audio_onnx)/len(audio_pt):.2f}x")


if __name__ == '__main__':
    main()
