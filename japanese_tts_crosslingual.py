#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japanese TTS using cross-lingual mode
Uses Chinese prompt audio to synthesize Japanese speech

This is the correct approach based on CosyVoice README
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

print("="*60)
print("CosyVoice2 Japanese TTS - Cross-Lingual Mode")
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

# Load Chinese prompt audio (16kHz)
print("Loading Chinese prompt audio...")
prompt_speech_16k = load_wav('chinese_prompt_16k.wav', 16000)
print(f"✓ Prompt loaded: {prompt_speech_16k.shape[1] / 16000:.2f}s")
print()

# Test texts with Japanese language tag
test_texts = [
    "<|jp|>今日は良い天気です。",
    "<|jp|>こんにちは、世界！",
    "<|jp|>明日も晴れるでしょう。",
]

print("Generating Japanese speech using cross-lingual mode...")
print("(Using Chinese voice with Japanese text)")
print()

for idx, text in enumerate(test_texts):
    print(f"[{idx+1}/{len(test_texts)}] Text: {text}")

    output_file = f"output_japanese_crosslingual_{idx+1}.wav"

    try:
        # Use inference_cross_lingual (correct method for multi-lingual)
        audio_chunks = []
        for i, chunk in enumerate(cosyvoice.inference_cross_lingual(
            text,
            prompt_speech_16k,
            stream=False,
            speed=1.0,
            text_frontend=False  # Use model's default processing
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
print("This uses the CORRECT approach:")
print("1. Chinese prompt audio (native voice)")
print("2. Japanese text with <|jp|> tag")
print("3. inference_cross_lingual() method")
print("4. text_frontend=False")
print()
print("This should produce proper Japanese pronunciation!")
