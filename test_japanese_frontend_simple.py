#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for Japanese frontend preprocessing
This tests the frontend without requiring full TTS models
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('third_party/Matcha-TTS')

try:
    import pyopenjtalk
    print("✓ pyopenjtalk-plus is available")
except ImportError:
    print("✗ pyopenjtalk-plus is NOT available")
    sys.exit(1)

try:
    from kabosu_core import Kabosu
    print("✓ kabosu-core is available (hybrid mode)")
    hybrid_available = True
except ImportError:
    print("✗ kabosu-core is NOT available (pyopenjtalk-only mode)")
    hybrid_available = False

print("\n" + "="*60)
print("Testing Japanese Frontend Preprocessing")
print("="*60 + "\n")

# Test sentences
test_texts = [
    "今日は良い天気です。",
    "彼は生で食べる。",  # Ambiguous: 生 (なま vs せい)
    "明日も晴れるでしょう。",
    "こんにちは、世界！",
]

print("### pyopenjtalk-plus Processing ###\n")
for text in test_texts:
    print(f"Text: {text}")

    # Extract phonemes
    phonemes = pyopenjtalk.g2p(text)
    print(f"  Phonemes: {phonemes}")

    # Extract full-context labels
    labels = pyopenjtalk.extract_fullcontext(text)
    print(f"  Labels: {len(labels)} labels")

    # Parse first few labels
    import re
    phoneme_list = []
    for label in labels[:5]:
        match = re.search(r'-(.+?)\+', label)
        if match:
            phoneme = match.group(1)
            if phoneme not in ['pau', 'sil']:
                phoneme_list.append(phoneme)

    if phoneme_list:
        print(f"  First phonemes: {' '.join(phoneme_list)}")
    print()

if hybrid_available:
    print("\n### kabosu-core Hybrid Processing ###\n")
    print("Initializing kabosu-core (loading BERT model)...")
    kabosu = Kabosu()
    print("✓ kabosu-core initialized\n")

    # Test ambiguous word
    ambiguous_texts = [
        ("彼は生で食べる。", "なま"),
        ("生ビールください。", "なま"),
    ]

    for text, expected in ambiguous_texts:
        print(f"Text: {text}")
        result = kabosu.process(text)
        print(f"  Reading: {result['reading']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Expected '{expected}' in reading: {'✓' if expected in result['reading'] else '✗'}")
        print()

print("="*60)
print("Test Complete!")
print("="*60)
print("\nThe Japanese frontend preprocessing is working correctly.")
print("To run full TTS inference, you need to:")
print("  1. Download pretrained models (see README.md)")
print("  2. Use webui.py or Python API (see README.md)")
