#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download pretrained CosyVoice models using modelscope

This script downloads the CosyVoice2-0.5B model (recommended for best quality)
"""

import os

print("="*60)
print("CosyVoice Model Download")
print("="*60)
print()

# Check if modelscope is installed
try:
    from modelscope import snapshot_download
    print("✓ modelscope is installed")
except ImportError:
    print("✗ modelscope is NOT installed")
    print("  Installing modelscope...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'modelscope'])
    from modelscope import snapshot_download
    print("✓ modelscope installed successfully")

print()
print("Downloading CosyVoice2-0.5B model...")
print("This will download several GB of data. Please be patient.")
print()

# Create pretrained_models directory
os.makedirs('pretrained_models', exist_ok=True)

# Download CosyVoice2-0.5B (recommended)
print("Downloading CosyVoice2-0.5B...")
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
print("✓ CosyVoice2-0.5B downloaded")

print()
print("="*60)
print("Download Complete!")
print("="*60)
print()
print("You can now run inference using:")
print("  python webui.py --model_dir pretrained_models/CosyVoice2-0.5B")
print()
print("Or create a custom inference script (see README.md for examples)")
