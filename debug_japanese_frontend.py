#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Japanese frontend processing to see what's happening
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
import pyopenjtalk

print("="*60)
print("Debug Japanese Frontend Processing")
print("="*60)
print()

# Test text
text = "今日は良い天気です。"
print(f"Input text: {text}")
print()

# Test pyopenjtalk directly
print("### Direct pyopenjtalk test ###")
phonemes = pyopenjtalk.g2p(text)
print(f"Phonemes: {phonemes}")

labels = pyopenjtalk.extract_fullcontext(text)
print(f"Full-context labels: {len(labels)} labels")
print("First 3 labels:")
for i, label in enumerate(labels[:3]):
    print(f"  {i+1}: {label[:100]}...")
print()

# Test CosyVoice frontend
print("### CosyVoice2 Frontend Test ###")
print("Loading model...")
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False
)
print("Model loaded")
print()

# Enable Japanese frontend
print("Enabling Japanese frontend...")
cosyvoice.frontend.use_japanese_frontend = True
cosyvoice.frontend.use_hybrid = False
print(f"use_japanese_frontend: {cosyvoice.frontend.use_japanese_frontend}")
print(f"use_hybrid: {cosyvoice.frontend.use_hybrid}")
print()

# Test text normalization
print("### Text Normalization Test ###")
normalized = cosyvoice.frontend.text_normalize(text, split=True, text_frontend=True)
print(f"Normalized text: {normalized}")
print()

# Check if Japanese processing is being used
print("### Japanese Detection Test ###")
is_japanese = cosyvoice.frontend._is_japanese(text)
print(f"Is Japanese detected: {is_japanese}")
print()

# Test with text_frontend=False to see difference
print("### Inference with text_frontend=True (Japanese processing) ###")
prompt_text = "你好"
prompt_speech = torch.zeros(1, int(cosyvoice.sample_rate * 0.5))

audio_chunks = []
for i, chunk in enumerate(cosyvoice.inference_zero_shot(
    text,
    prompt_text,
    prompt_speech,
    stream=False,
    speed=1.0,
    text_frontend=True  # Enable Japanese frontend
)):
    audio_chunks.append(chunk['tts_speech'])
    print(f"Generated chunk {i+1}")

if audio_chunks:
    audio = torch.cat(audio_chunks, dim=1)
    torchaudio.save("debug_with_japanese.wav", audio, cosyvoice.sample_rate)
    print(f"✓ Saved: debug_with_japanese.wav")
print()

print("### Inference with text_frontend=False (No Japanese processing) ###")
audio_chunks = []
for i, chunk in enumerate(cosyvoice.inference_zero_shot(
    text,
    prompt_text,
    prompt_speech,
    stream=False,
    speed=1.0,
    text_frontend=False  # Disable Japanese frontend
)):
    audio_chunks.append(chunk['tts_speech'])
    print(f"Generated chunk {i+1}")

if audio_chunks:
    audio = torch.cat(audio_chunks, dim=1)
    torchaudio.save("debug_without_japanese.wav", audio, cosyvoice.sample_rate)
    print(f"✓ Saved: debug_without_japanese.wav")
print()

print("="*60)
print("Debug Complete!")
print("="*60)
print()
print("Compare the two audio files:")
print("  - debug_with_japanese.wav (Japanese frontend enabled)")
print("  - debug_without_japanese.wav (Japanese frontend disabled)")
print()
print("If Japanese frontend is working correctly, 'with_japanese' should")
print("sound better with correct Japanese pronunciation.")
