#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct Japanese TTS Demo using CosyVoice2 with language tags

Key fixes:
1. Use <|jp|> language tag for Japanese text
2. Set text_frontend=False (as recommended in README)
3. Use proper zero-shot inference

Usage:
    python simple_japanese_tts_fixed.py
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

print("="*60)
print("CosyVoice2 Japanese TTS Demo (FIXED)")
print("="*60)
print()

# Initialize CosyVoice2
print("Loading CosyVoice2 model...")
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False
)
print(f"✓ Model loaded (sample_rate: {cosyvoice.sample_rate} Hz)")
print()

# Test texts with Japanese language tag
test_texts = [
    "<|jp|>今日は良い天気です。",
    "<|jp|>こんにちは、世界！",
    "<|jp|>明日も晴れるでしょう。",
]

print("Generating speech for Japanese text...")
print("Using <|jp|> language tag and text_frontend=False")
print()

# Create a simple prompt (can use silence or a real audio file)
# For cross-lingual, we can use Chinese prompt with Japanese target
prompt_text = "你好"  # Chinese prompt
prompt_speech = torch.zeros(1, int(cosyvoice.sample_rate * 0.5))  # 0.5 second silence

for idx, text in enumerate(test_texts):
    print(f"[{idx+1}/{len(test_texts)}] Text: {text}")

    output_file = f"output_japanese_fixed_{idx+1}.wav"

    try:
        # Generate with zero-shot, text_frontend=False
        audio_chunks = []
        for i, chunk in enumerate(cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            prompt_speech,
            stream=False,
            speed=1.0,
            text_frontend=False  # KEY: Use False as recommended
        )):
            audio_chunks.append(chunk['tts_speech'])

        if audio_chunks:
            audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(output_file, audio, cosyvoice.sample_rate)
            duration = audio.shape[1] / cosyvoice.sample_rate
            print(f"  ✓ Saved to: {output_file} ({duration:.2f}s)")
        else:
            print(f"  ✗ No audio generated")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print()

print("="*60)
print("Demo Complete!")
print("="*60)
print()
print("Key changes from previous version:")
print("1. Added <|jp|> language tag to Japanese text")
print("2. Set text_frontend=False (as recommended in README)")
print("3. This should produce correct Japanese pronunciation")
