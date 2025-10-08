#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRECT Japanese TTS using CosyVoice2

This script demonstrates the CORRECT way to use Japanese TTS:
1. Convert Japanese text (with kanji) to hiragana using pyopenjtalk
2. Add <|jp|> language tag
3. Use text_frontend=False (as recommended in README)

This approach resolves kanji reading ambiguities before passing to the model.

Usage:
    python simple_japanese_tts_correct.py
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.japanese_utils import prepare_japanese_text_for_tts

print("="*60)
print("CosyVoice2 Japanese TTS - CORRECT Implementation")
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

# Test texts - original Japanese with kanji
test_texts = [
    "今日は良い天気です。",
    "こんにちは、世界！",
    "明日も晴れるでしょう。",
]

print("="*60)
print("CORRECT APPROACH:")
print("1. Convert kanji to hiragana using pyopenjtalk")
print("2. Add <|jp|> language tag")
print("3. Use text_frontend=False")
print("="*60)
print()

for idx, text in enumerate(test_texts):
    print(f"[{idx+1}/{len(test_texts)}]")
    print(f"  Original: {text}")

    # Convert Japanese text to hiragana (resolves kanji ambiguity)
    prepared_text = prepare_japanese_text_for_tts(text, add_language_tag=True)
    print(f"  Prepared: {prepared_text}")

    output_file = f"output_japanese_correct_{idx+1}.wav"

    try:
        # Use cross-lingual inference with text_frontend=False
        audio_chunks = []
        for i, chunk in enumerate(cosyvoice.inference_cross_lingual(
            prepared_text,  # ← hiragana text with <|jp|> tag
            prompt_speech_16k,
            stream=False,
            speed=1.0,
            text_frontend=False  # ← No text processing by model
        )):
            audio_chunks.append(chunk['tts_speech'])

        if audio_chunks:
            audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(output_file, audio, cosyvoice.sample_rate)
            duration = audio.shape[1] / cosyvoice.sample_rate
            print(f"  ✓ Saved: {output_file} ({duration:.2f}s)")
        else:
            print(f"  ✗ No audio generated")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print()

print("="*60)
print("Generation Complete!")
print("="*60)
print()
print("Key points of this CORRECT implementation:")
print("1. Uses pyopenjtalk.g2p() to convert kanji to phonemes")
print("2. Converts phonemes to hiragana (unambiguous representation)")
print("3. Adds <|jp|> language tag")
print("4. Uses text_frontend=False (as recommended)")
print()
print("This resolves kanji reading ambiguities BEFORE passing to the model,")
print("ensuring correct Japanese pronunciation.")
print()
print("Compare these audio files with previous attempts:")
print("  - output_japanese_correct_1.wav")
print("  - output_japanese_correct_2.wav")
print("  - output_japanese_correct_3.wav")
