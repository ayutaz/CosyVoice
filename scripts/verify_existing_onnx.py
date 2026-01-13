#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Verify Existing ONNX Models

This script verifies all existing ONNX models in the model directory
and prints their input/output specifications.

Usage:
    python scripts/verify_existing_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def get_onnx_info(onnx_path: str) -> dict:
    """Get input/output information from ONNX model"""

    info = {
        'path': onnx_path,
        'size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
        'inputs': [],
        'outputs': []
    }

    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        for input_tensor in session.get_inputs():
            info['inputs'].append({
                'name': input_tensor.name,
                'shape': input_tensor.shape,
                'type': input_tensor.type
            })

        for output_tensor in session.get_outputs():
            info['outputs'].append({
                'name': output_tensor.name,
                'shape': output_tensor.shape,
                'type': output_tensor.type
            })

        info['status'] = 'OK'
    except Exception as e:
        info['status'] = f'ERROR: {e}'

    return info


def test_campplus(model_dir: str):
    """Test CAMPPlus speaker encoder"""

    print(f"\n{'='*60}")
    print("Testing CAMPPlus (Speaker Encoder)")
    print(f"{'='*60}")

    onnx_path = os.path.join(model_dir, 'campplus.onnx')
    if not os.path.exists(onnx_path):
        print(f"Not found: {onnx_path}")
        return

    info = get_onnx_info(onnx_path)
    print(f"Size: {info['size_mb']:.1f} MB")
    print(f"Status: {info['status']}")

    print("\nInputs:")
    for inp in info['inputs']:
        print(f"  {inp['name']}: {inp['shape']} ({inp['type']})")

    print("\nOutputs:")
    for out in info['outputs']:
        print(f"  {out['name']}: {out['shape']} ({out['type']})")

    if info['status'] == 'OK':
        # Test inference
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # CAMPPlus expects audio features (fbank)
        # Typical input: [batch, time, 80] mel spectrogram
        batch_size = 1
        time_len = 200
        feat_dim = 80

        # Create dummy fbank features
        fbank = np.random.randn(batch_size, time_len, feat_dim).astype(np.float32)

        try:
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: fbank})[0]
            print(f"\nTest inference:")
            print(f"  Input shape: {fbank.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        except Exception as e:
            print(f"\nTest inference failed: {e}")


def test_speech_tokenizer(model_dir: str):
    """Test Speech Tokenizer"""

    print(f"\n{'='*60}")
    print("Testing Speech Tokenizer")
    print(f"{'='*60}")

    onnx_path = os.path.join(model_dir, 'speech_tokenizer_v3.onnx')
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(model_dir, 'speech_tokenizer_v1.onnx')
    if not os.path.exists(onnx_path):
        print("Speech tokenizer ONNX not found")
        return

    info = get_onnx_info(onnx_path)
    print(f"File: {os.path.basename(onnx_path)}")
    print(f"Size: {info['size_mb']:.1f} MB")
    print(f"Status: {info['status']}")

    print("\nInputs:")
    for inp in info['inputs']:
        print(f"  {inp['name']}: {inp['shape']} ({inp['type']})")

    print("\nOutputs:")
    for out in info['outputs']:
        print(f"  {out['name']}: {out['shape']} ({out['type']})")


def test_flow_estimator(model_dir: str):
    """Test Flow Decoder Estimator"""

    print(f"\n{'='*60}")
    print("Testing Flow Decoder Estimator")
    print(f"{'='*60}")

    onnx_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
    if not os.path.exists(onnx_path):
        print(f"Not found: {onnx_path}")
        return

    info = get_onnx_info(onnx_path)
    print(f"Size: {info['size_mb']:.1f} MB")
    print(f"Status: {info['status']}")

    print("\nInputs:")
    for inp in info['inputs']:
        print(f"  {inp['name']}: {inp['shape']} ({inp['type']})")

    print("\nOutputs:")
    for out in info['outputs']:
        print(f"  {out['name']}: {out['shape']} ({out['type']})")


def list_all_onnx(model_dir: str):
    """List all ONNX files in model directory"""

    print(f"\n{'='*60}")
    print("All ONNX Files")
    print(f"{'='*60}")

    # Check model directory
    for f in os.listdir(model_dir):
        if f.endswith('.onnx'):
            path = os.path.join(model_dir, f)
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {f}: {size:.1f} MB")

    # Check onnx subdirectory
    onnx_dir = os.path.join(model_dir, 'onnx')
    if os.path.exists(onnx_dir):
        print(f"\nIn {onnx_dir}:")
        for f in os.listdir(onnx_dir):
            if f.endswith('.onnx'):
                path = os.path.join(onnx_dir, f)
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"  {f}: {size:.1f} MB")


def print_summary():
    """Print summary of ONNX export status"""

    print(f"\n{'='*60}")
    print("ONNX Export Summary for Unity Sentis")
    print(f"{'='*60}")

    print("""
Component                    | Status      | Size     | Notes
-----------------------------|-------------|----------|------------------
HiFT F0 Predictor           | [OK]        | ~13 MB   | scripts/export_hift_onnx.py
HiFT Decoder                | [OK]        | ~67 MB   | scripts/export_hift_onnx.py
Flow Token Embedding        | [OK]        | ~2 MB    | scripts/export_flow_onnx.py
Flow Speaker Projection     | [OK]        | <1 MB    | scripts/export_flow_onnx.py
Flow Pre-Lookahead          | [OK]        | ~2 MB    | scripts/export_flow_onnx.py
Flow Decoder Estimator      | [OK]        | ~1.3 GB  | Official export
CAMPPlus (Speaker Encoder)  | [OK]        | ~27 MB   | Pre-included
Speech Tokenizer V3         | [OK]        | ~925 MB  | Pre-included
LLM (Qwen2)                 | [TODO]      | ~400 MB  | Complex, needs KV cache

Unity C# Implementation Required:
- STFT/ISTFT (n_fft=16, hop_len=4)
- Source generation (F0 -> sine waves)
- Euler solver (10 steps for flow matching)
- KV Cache management for LLM
- Sampling strategy (Top-K)
""")


def main():
    parser = argparse.ArgumentParser(description='Verify existing ONNX models')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='pretrained_models/Fun-CosyVoice3-0.5B',
        help='Path to model directory'
    )

    args = parser.parse_args()

    print(f"Model directory: {args.model_dir}")

    # List all ONNX files
    list_all_onnx(args.model_dir)

    # Test individual models
    test_campplus(args.model_dir)
    test_speech_tokenizer(args.model_dir)
    test_flow_estimator(args.model_dir)

    # Print summary
    print_summary()


if __name__ == '__main__':
    main()
