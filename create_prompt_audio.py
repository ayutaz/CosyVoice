#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a simple audio prompt for testing
Generate a Chinese audio first, then use it for cross-lingual Japanese
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

print("Creating Chinese audio prompt for cross-lingual Japanese synthesis...")

cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False
)

# Generate a Chinese audio as prompt
chinese_text = "你好，我是语音合成系统。"
prompt_text = "你好"
prompt_speech = torch.zeros(1, int(cosyvoice.sample_rate * 0.5))

print(f"Generating Chinese prompt: {chinese_text}")

audio_chunks = []
for i, chunk in enumerate(cosyvoice.inference_zero_shot(
    chinese_text,
    prompt_text,
    prompt_speech,
    stream=False,
    speed=1.0,
    text_frontend=False
)):
    audio_chunks.append(chunk['tts_speech'])

if audio_chunks:
    audio = torch.cat(audio_chunks, dim=1)
    # Resample to 16kHz for use as prompt
    resampler = torchaudio.transforms.Resample(
        orig_freq=cosyvoice.sample_rate,
        new_freq=16000
    )
    audio_16k = resampler(audio)
    torchaudio.save("chinese_prompt_16k.wav", audio_16k, 16000)
    print(f"✓ Saved Chinese prompt: chinese_prompt_16k.wav")
    print(f"  Duration: {audio_16k.shape[1] / 16000:.2f}s")
else:
    print("✗ Failed to generate prompt")
