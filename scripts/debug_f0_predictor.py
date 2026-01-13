#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Debug F0 Predictor ONNX vs PyTorch differences.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

from hyperpyyaml import load_hyperpyyaml

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
ONNX_DIR = os.path.join(MODEL_DIR, 'onnx')


def load_hift_model():
    """Load HiFT model"""
    yaml_path = os.path.join(MODEL_DIR, 'cosyvoice3.yaml')
    with open(yaml_path, 'r') as f:
        qwen_path = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path})

    hift = configs['hift']
    hift_state = {k.replace('generator.', ''): v
                  for k, v in torch.load(os.path.join(MODEL_DIR, 'hift.pt'), map_location='cpu').items()}
    hift.load_state_dict(hift_state)
    hift.eval()
    return hift


def main():
    print("=" * 60)
    print("F0 Predictor Debug")
    print("=" * 60)

    hift = load_hift_model()
    f0_predictor = hift.f0_predictor

    # Load ONNX session
    onnx_path = os.path.join(ONNX_DIR, 'hift_f0_predictor_fp16.onnx')
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(ONNX_DIR, 'hift_f0_predictor_fp32.onnx')

    print(f"Loading ONNX from: {onnx_path}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    onnx_session = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])

    # Check ONNX inputs/outputs
    print("\nONNX Model Info:")
    for inp in onnx_session.get_inputs():
        print(f"  Input: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
    for out in onnx_session.get_outputs():
        print(f"  Output: {out.name}, Shape: {out.shape}, Type: {out.type}")

    # Check F0 Predictor structure
    print("\nPyTorch F0 Predictor Structure:")
    print(f"  Type: {type(f0_predictor)}")
    print(f"  condnet layers: {len(f0_predictor.condnet)}")

    for i, layer in enumerate(f0_predictor.condnet):
        if hasattr(layer, 'causal_padding'):
            print(f"    Layer {i}: {type(layer).__name__}, causal_padding={layer.causal_padding}")
        else:
            print(f"    Layer {i}: {type(layer).__name__}")

    # Test with specific input
    print("\n" + "=" * 60)
    print("Test 1: Random input")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    mel = np.random.randn(1, 80, 50).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = f0_predictor(torch.from_numpy(mel), finalize=True)
        pt_out_np = pt_out.numpy()

    # ONNX
    onnx_out = onnx_session.run(None, {'mel': mel})[0]

    print(f"PyTorch output: shape={pt_out_np.shape}, range=[{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")
    print(f"ONNX output: shape={onnx_out.shape}, range=[{onnx_out.min():.4f}, {onnx_out.max():.4f}]")

    diff = np.abs(pt_out_np - onnx_out)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    # Check layer by layer
    print("\n" + "=" * 60)
    print("Layer-by-layer comparison")
    print("=" * 60)

    # Run through condnet step by step
    x_pt = torch.from_numpy(mel)

    print("\nPyTorch intermediate outputs:")
    for i, layer in enumerate(f0_predictor.condnet):
        with torch.no_grad():
            if hasattr(layer, 'causal_padding'):
                # CausalConv1d layer
                x_pt = layer(x_pt)
                print(f"  After layer {i} ({type(layer).__name__}): shape={x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")
            else:
                # Activation layer
                x_pt = layer(x_pt)
                print(f"  After layer {i} ({type(layer).__name__}): shape={x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")

    # Classifier
    with torch.no_grad():
        x_pt = x_pt.transpose(1, 2)
        x_pt = torch.abs(f0_predictor.classifier(x_pt).squeeze(-1))
        print(f"  After classifier: shape={x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")

    # Test 2: Check different precision
    print("\n" + "=" * 60)
    print("Test 2: Check FP32 vs FP16")
    print("=" * 60)

    # Try FP32 model
    fp32_path = os.path.join(ONNX_DIR, 'hift_f0_predictor_fp32.onnx')
    if os.path.exists(fp32_path):
        fp32_session = ort.InferenceSession(fp32_path, so, providers=['CPUExecutionProvider'])
        fp32_out = fp32_session.run(None, {'mel': mel})[0]
        print(f"FP32 output: shape={fp32_out.shape}, range=[{fp32_out.min():.4f}, {fp32_out.max():.4f}]")

        diff_fp32 = np.abs(pt_out_np - fp32_out)
        print(f"FP32 Max diff from PyTorch: {diff_fp32.max():.6f}")
        print(f"FP32 Mean diff from PyTorch: {diff_fp32.mean():.6f}")

    # Test 3: Check first layer behavior
    print("\n" + "=" * 60)
    print("Test 3: First CausalConv1d layer analysis")
    print("=" * 60)

    first_layer = f0_predictor.condnet[0]
    print(f"First layer type: {type(first_layer).__name__}")
    print(f"  kernel_size: {first_layer.kernel_size}")
    print(f"  causal_padding: {first_layer.causal_padding}")
    print(f"  causal_type: {first_layer.causal_type}")
    print(f"  weight shape: {first_layer.weight.shape}")
    if first_layer.bias is not None:
        print(f"  bias shape: {first_layer.bias.shape}")

    # Test with simple input
    simple_input = np.ones((1, 80, 10), dtype=np.float32) * 0.5

    with torch.no_grad():
        simple_pt = f0_predictor(torch.from_numpy(simple_input), finalize=True).numpy()
    simple_onnx = onnx_session.run(None, {'mel': simple_input})[0]

    print(f"\nSimple input test (all 0.5):")
    print(f"  PyTorch: {simple_pt}")
    print(f"  ONNX: {simple_onnx}")
    print(f"  Max diff: {np.abs(simple_pt - simple_onnx).max():.6f}")

    # Test 4: Check if it's a cache issue
    print("\n" + "=" * 60)
    print("Test 4: Manual forward pass comparison")
    print("=" * 60)

    # Run first conv manually with explicit cache
    with torch.no_grad():
        x = torch.from_numpy(mel)

        # Manual first layer with explicit cache
        cache = torch.zeros(x.shape[0], x.shape[1], first_layer.causal_padding)

        if first_layer.causal_type == 'left':
            padded = torch.cat([cache, x], dim=2)
        else:
            padded = torch.cat([x, cache], dim=2)

        print(f"Input shape: {x.shape}")
        print(f"Cache shape: {cache.shape}")
        print(f"Padded shape: {padded.shape}")

        # Apply convolution using parent forward
        conv_out = torch.nn.functional.conv1d(
            padded,
            first_layer.weight,
            first_layer.bias,
            stride=first_layer.stride,
            padding=first_layer.padding,
            dilation=first_layer.dilation,
            groups=first_layer.groups
        )
        print(f"Conv output shape: {conv_out.shape}")

    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)

    if diff.max() > 0.5:
        print("FAIL: Significant difference between PyTorch and ONNX F0 Predictor")
        print("\nPossible causes:")
        print("1. CausalConv1d default cache handling differs in ONNX")
        print("2. FP16 precision loss")
        print("3. Export wrapper not correctly handling finalize parameter")
    else:
        print("PASS: F0 Predictor outputs are close")


if __name__ == '__main__':
    main()
