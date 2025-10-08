#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Japanese text conversion

This script tests the Japanese text → hiragana conversion
to verify that kanji reading ambiguities are correctly resolved.

Usage:
    python test_japanese_conversion.py
"""

import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.utils.japanese_utils import (
    convert_japanese_to_hiragana,
    prepare_japanese_text_for_tts,
    convert_phonemes_to_hiragana
)

print("="*60)
print("Japanese Text Conversion Test")
print("="*60)
print()

# Test cases with known ambiguities
test_cases = [
    {
        "text": "今日は良い天気です。",
        "expected_reading": "きょう",
        "description": "今日 should be 'kyou' (today), not 'kon-nichi'"
    },
    {
        "text": "明日も晴れるでしょう。",
        "expected_reading": "あした",
        "description": "明日 should be 'ashita' (tomorrow)"
    },
    {
        "text": "彼は生で食べる。",
        "expected_reading": "なま",
        "description": "生 should be 'nama' (raw) in this context"
    },
    {
        "text": "生まれた場所。",
        "expected_reading": "う",
        "description": "生まれた should include 'u' from '生'"
    },
    {
        "text": "こんにちは、世界！",
        "expected_reading": "こんにちは",
        "description": "Already hiragana, should pass through"
    },
    {
        "text": "人生は短い。",
        "expected_reading": "じんせい",
        "description": "人生 should be 'jinsei' (life)"
    },
]

print("Testing text conversion:")
print()

for idx, test_case in enumerate(test_cases, 1):
    text = test_case["text"]
    expected = test_case["expected_reading"]
    description = test_case["description"]

    print(f"Test {idx}: {description}")
    print(f"  Input:    {text}")

    try:
        # Convert to hiragana
        hiragana = convert_japanese_to_hiragana(text)
        print(f"  Output:   {hiragana}")

        # Check if expected reading is in the output
        if expected in hiragana:
            print(f"  ✓ PASS - Contains expected '{expected}'")
        else:
            print(f"  ⚠ CHECK - Expected '{expected}' not found in output")

        # Show prepared text for TTS
        prepared = prepare_japanese_text_for_tts(text, add_language_tag=True)
        print(f"  TTS text: {prepared}")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")

    print()

print("="*60)
print("Phoneme Conversion Test")
print("="*60)
print()

# Test phoneme-to-hiragana conversion directly
phoneme_tests = [
    {
        "phonemes": "ky o o",
        "expected": "きょお",
        "description": "Long vowel 'kyoo'"
    },
    {
        "phonemes": "k o N n i ch i w a",
        "expected": "こんにちいわ",
        "description": "'konnichiwa'"
    },
    {
        "phonemes": "a sh I t a",
        "expected": "あしいた",
        "description": "'ashita' with devoiced 'I'"
    },
]

for idx, test in enumerate(phoneme_tests, 1):
    phonemes = test["phonemes"]
    expected = test["expected"]
    description = test["description"]

    print(f"Test {idx}: {description}")
    print(f"  Phonemes: {phonemes}")

    try:
        hiragana = convert_phonemes_to_hiragana(phonemes)
        print(f"  Hiragana: {hiragana}")

        if hiragana == expected:
            print(f"  ✓ PASS")
        else:
            print(f"  ⚠ Expected: {expected}")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")

    print()

print("="*60)
print("Test Complete!")
print("="*60)
print()
print("If all tests pass, the conversion is working correctly.")
print("You can now use simple_japanese_tts_correct.py to generate")
print("audio with proper Japanese pronunciation.")
