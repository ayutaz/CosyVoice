#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Japanese TTS Demo using CosyVoice2
Uses zero-shot mode with a reference audio

Usage:
    python simple_japanese_tts.py
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

print("="*60)
print("CosyVoice2 Japanese TTS Demo")
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

# Enable Japanese frontend
print("Enabling Japanese frontend preprocessing...")
cosyvoice.frontend.use_japanese_frontend = True
cosyvoice.frontend.use_hybrid = False  # pyopenjtalk-only mode
print("✓ Japanese frontend enabled (pyopenjtalk-only mode)")
print()

# Test texts
test_texts = [
    "今日は良い天気です。",
    "こんにちは、世界！",
    "明日も晴れるでしょう。",
]

print("Generating speech for Japanese text...")
print()

# Use zero-shot mode (no prompt needed for simple generation)
# For CosyVoice2, we can generate with default settings
for idx, text in enumerate(test_texts):
    print(f"[{idx+1}/{len(test_texts)}] Text: {text}")

    # Generate audio using zero-shot mode without prompt
    # This will use default voice characteristics
    output_file = f"output_japanese_{idx+1}.wav"

    try:
        # Create a simple prompt
        prompt_text = "你好"
        prompt_speech = torch.zeros(1, int(cosyvoice.sample_rate * 0.5))  # 0.5 second silence as dummy prompt

        # Generate with zero-shot
        audio_chunks = []
        for i, chunk in enumerate(cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            prompt_speech,
            stream=False,
            speed=1.0
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
print("Notes:")
print("- The Japanese frontend (pyopenjtalk-plus) is being used for text preprocessing")
print("- This improves phoneme and accent accuracy for Japanese text")
print("- For better quality, you can provide a reference audio prompt")
