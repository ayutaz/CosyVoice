#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity Sentis ONNX Export
# Apache License 2.0

"""
Unity Sentis Compatible ONNX Export Script

This script re-exports CosyVoice3 models with opset version 15 for Unity Sentis compatibility.

Key differences from standard export:
- Uses opset version 15 (required for Sentis, supported range: 7-15)
- Verifies Sentis compatibility
- Reports unsupported operators

Models to re-export:
1. flow.decoder.estimator (opset 18 → 15)
2. hift_source_generator (opset 17 → 15)
3. text_embedding (opset 17 → 15)

Usage:
    python scripts/export_onnx_sentis.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Set, List, Dict

import numpy as np
import torch
import torch.nn as nn
import onnx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

from hyperpyyaml import load_hyperpyyaml

# Unity Sentis configuration
SENTIS_OPSET_VERSION = 15  # Required for Unity Sentis (7-15)
SENTIS_MAX_TENSOR_DIMS = 8  # Maximum tensor dimensions

# Operators known to be unsupported by Unity Sentis
SENTIS_UNSUPPORTED_OPS: Set[str] = {
    "If",  # Conditional control flow
    "Log1p",  # Use Log(1+x) instead
    "LogAddExp",
    "ComplexAbs",
    "FFT", "IFFT", "RFFT", "IRFFT",  # Signal processing
    "Unique",
    "StringNormalizer", "TfIdfVectorizer", "Tokenizer",
    "GRU", "RNN", "Loop",  # Recurrent/control flow
    "QuantizeLinear", "DequantizeLinear",  # Quantization (removed in recent versions)
}


def verify_sentis_compatibility(model_path: str) -> List[str]:
    """Verify that the ONNX model is compatible with Unity Sentis.

    Returns list of warnings/errors.
    """
    model = onnx.load(model_path)
    issues = []

    # Check opset version
    opset_version = model.opset_import[0].version if model.opset_import else 0
    if opset_version > SENTIS_OPSET_VERSION:
        issues.append(f"ERROR: opset version {opset_version} > {SENTIS_OPSET_VERSION} (Sentis limit)")

    # Check for unsupported operators
    ops_used = set()
    for node in model.graph.node:
        ops_used.add(node.op_type)
        if node.op_type in SENTIS_UNSUPPORTED_OPS:
            issues.append(f"ERROR: Operator '{node.op_type}' (node: {node.name}) is not supported by Sentis")

    # Check tensor dimensions
    for value_info in model.graph.value_info:
        if value_info.type.HasField("tensor_type"):
            dims = len(value_info.type.tensor_type.shape.dim)
            if dims > SENTIS_MAX_TENSOR_DIMS:
                issues.append(f"ERROR: Tensor '{value_info.name}' has {dims} dims > {SENTIS_MAX_TENSOR_DIMS}")

    for io_info in list(model.graph.input) + list(model.graph.output):
        if io_info.type.HasField("tensor_type"):
            dims = len(io_info.type.tensor_type.shape.dim)
            if dims > SENTIS_MAX_TENSOR_DIMS:
                issues.append(f"ERROR: I/O tensor '{io_info.name}' has {dims} dims > {SENTIS_MAX_TENSOR_DIMS}")

    return issues, ops_used


def add_sentis_metadata(filename: str, meta_data: Dict[str, str]):
    """Add metadata to ONNX model for Sentis."""
    model = onnx.load(filename)

    # Add target runtime info
    meta_data["target_runtime"] = "unity_sentis"
    meta_data["opset_version"] = str(SENTIS_OPSET_VERSION)

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class TextEmbeddingWrapper(nn.Module):
    """Wrapper for Qwen2 text embedding layer"""

    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class SourceGeneratorWrapper(nn.Module):
    """Wrapper for HiFT source generation"""

    def __init__(self, f0_upsamp, m_source):
        super().__init__()
        self.f0_upsamp = f0_upsamp
        self.m_source = m_source

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        # f0: [batch, 1, time]
        s = self.f0_upsamp(f0).transpose(1, 2)
        source, _, _ = self.m_source(s)
        return source.transpose(1, 2)


def export_flow_estimator_sentis(model_dir: str, output_dir: str, device: str = 'cpu'):
    """Export Flow Decoder Estimator with opset 15"""
    print("\n" + "=" * 60)
    print("Exporting Flow Decoder Estimator (opset 15 for Sentis)")
    print("=" * 60)

    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')

    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'hift': None})

    flow = configs['flow']

    # Load weights
    flow_path = os.path.join(model_dir, 'flow.pt')
    if os.path.exists(flow_path):
        print(f"Loading weights from {flow_path}")
        state_dict = torch.load(flow_path, map_location=device)
        flow.load_state_dict(state_dict)

    flow.eval()
    flow.to(device)

    estimator = flow.decoder.estimator

    # Dummy inputs
    batch_size, seq_len = 2, 256
    out_channels = estimator.out_channels

    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)

    output_path = os.path.join(output_dir, 'flow.decoder.estimator.sentis.onnx')

    print(f"Exporting to {output_path} with opset {SENTIS_OPSET_VERSION}...")

    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        output_path,
        export_params=True,
        opset_version=SENTIS_OPSET_VERSION,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes={
            'x': {2: 'seq_len'},
            'mask': {2: 'seq_len'},
            'mu': {2: 'seq_len'},
            'cond': {2: 'seq_len'},
            'estimator_out': {2: 'seq_len'},
        }
    )

    # Add metadata
    add_sentis_metadata(output_path, {
        "model_type": "flow_decoder_estimator",
        "out_channels": str(out_channels),
    })

    # Verify
    issues, ops = verify_sentis_compatibility(output_path)
    print(f"Operators used: {sorted(ops)}")

    if issues:
        print("\nSentis compatibility issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("Sentis compatibility: PASSED")

    # File size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    return output_path


def export_text_embedding_sentis(model_dir: str, output_dir: str, device: str = 'cpu'):
    """Export Text Embedding with opset 15"""
    print("\n" + "=" * 60)
    print("Exporting Text Embedding (opset 15 for Sentis)")
    print("=" * 60)

    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')

    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path, 'flow': None, 'hift': None})

    llm = configs['llm']

    # Load weights
    llm_path = os.path.join(model_dir, 'llm.pt')
    if os.path.exists(llm_path):
        print(f"Loading weights from {llm_path}")
        state_dict = torch.load(llm_path, map_location=device)
        llm.load_state_dict(state_dict)

    llm.eval()

    embed_tokens = llm.llm.model.model.embed_tokens
    wrapper = TextEmbeddingWrapper(embed_tokens)
    wrapper.eval()

    dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    output_path = os.path.join(output_dir, 'text_embedding.sentis.onnx')

    print(f"Exporting to {output_path} with opset {SENTIS_OPSET_VERSION}...")

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['embeddings'],
        opset_version=SENTIS_OPSET_VERSION,
        do_constant_folding=True,
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'embeddings': {0: 'batch', 1: 'seq_len'}
        }
    )

    # Add metadata
    add_sentis_metadata(output_path, {
        "model_type": "text_embedding",
        "embed_dim": str(embed_tokens.embedding_dim),
        "vocab_size": str(embed_tokens.num_embeddings),
    })

    # Verify
    issues, ops = verify_sentis_compatibility(output_path)
    print(f"Operators used: {sorted(ops)}")

    if issues:
        print("\nSentis compatibility issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("Sentis compatibility: PASSED")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    return output_path


def export_source_generator_sentis(model_dir: str, output_dir: str, device: str = 'cpu'):
    """Export HiFT Source Generator with opset 15"""
    print("\n" + "=" * 60)
    print("Exporting HiFT Source Generator (opset 15 for Sentis)")
    print("=" * 60)

    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')

    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})

    hift = configs['hift']

    # Load weights
    hift_path = os.path.join(model_dir, 'hift.pt')
    if os.path.exists(hift_path):
        print(f"Loading weights from {hift_path}")
        state_dict = torch.load(hift_path, map_location=device)
        hift.load_state_dict(state_dict)

    hift.eval()
    hift.to(device)

    wrapper = SourceGeneratorWrapper(hift.f0_upsamp, hift.m_source)
    wrapper.eval()

    # Dummy input
    batch_size = 1
    mel_len = 100
    f0 = torch.randn(batch_size, 1, mel_len, device=device) * 200 + 200

    print(f"Input shape: f0 {f0.shape}")

    with torch.no_grad():
        output = wrapper(f0)
    print(f"Output shape: source {output.shape}")

    output_path = os.path.join(output_dir, 'hift_source_generator.sentis.onnx')

    print(f"Exporting to {output_path} with opset {SENTIS_OPSET_VERSION}...")

    torch.onnx.export(
        wrapper,
        (f0,),
        output_path,
        opset_version=SENTIS_OPSET_VERSION,
        do_constant_folding=True,
        input_names=['f0'],
        output_names=['source'],
        dynamic_axes={
            'f0': {0: 'batch', 2: 'mel_len'},
            'source': {0: 'batch', 2: 'source_len'}
        }
    )

    # Add metadata
    add_sentis_metadata(output_path, {
        "model_type": "hift_source_generator",
        "sample_rate": "24000",
    })

    # Verify
    issues, ops = verify_sentis_compatibility(output_path)
    print(f"Operators used: {sorted(ops)}")

    if issues:
        print("\nSentis compatibility issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("Sentis compatibility: PASSED")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    return output_path


def verify_all_models(output_dir: str):
    """Verify all ONNX models in directory for Sentis compatibility"""
    print("\n" + "=" * 60)
    print("Verifying All Models for Unity Sentis Compatibility")
    print("=" * 60)

    onnx_files = list(Path(output_dir).glob('*.onnx'))

    results = []
    for onnx_path in sorted(onnx_files):
        model = onnx.load(str(onnx_path))
        opset = model.opset_import[0].version if model.opset_import else 0

        issues, ops = verify_sentis_compatibility(str(onnx_path))

        status = "PASS" if not issues and opset <= SENTIS_OPSET_VERSION else "FAIL"

        results.append({
            'file': onnx_path.name,
            'opset': opset,
            'status': status,
            'issues': len(issues),
        })

        print(f"\n{onnx_path.name}:")
        print(f"  Opset: {opset} {'OK' if opset <= SENTIS_OPSET_VERSION else 'FAIL'}")
        print(f"  Status: {status}")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                print(f"  - {issue}")

    print("\n" + "-" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"{'File':<45} {'Opset':<8} {'Status'}")
    print("-" * 60)
    for r in results:
        print(f"{r['file']:<45} {r['opset']:<8} {r['status']}")


def main():
    parser = argparse.ArgumentParser(description='Export CosyVoice3 models for Unity Sentis')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='pretrained_models/Fun-CosyVoice3-0.5B',
        help='Path to model directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: model_dir/onnx_sentis)'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing models, do not export'
    )
    parser.add_argument(
        '--export_all',
        action='store_true',
        help='Export all components (flow_estimator, text_embedding, source_generator)'
    )
    parser.add_argument(
        '--export_flow',
        action='store_true',
        help='Export flow decoder estimator'
    )
    parser.add_argument(
        '--export_text',
        action='store_true',
        help='Export text embedding'
    )
    parser.add_argument(
        '--export_source',
        action='store_true',
        help='Export source generator'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'onnx_sentis')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sentis opset version: {SENTIS_OPSET_VERSION}")

    if args.verify_only:
        verify_all_models(os.path.join(args.model_dir, 'onnx'))
        return

    # Default to export_all if no specific export flag is set
    if not (args.export_flow or args.export_text or args.export_source):
        args.export_all = True

    if args.export_all or args.export_flow:
        export_flow_estimator_sentis(args.model_dir, args.output_dir)

    if args.export_all or args.export_text:
        export_text_embedding_sentis(args.model_dir, args.output_dir)

    if args.export_all or args.export_source:
        export_source_generator_sentis(args.model_dir, args.output_dir)

    # Verify all exported models
    verify_all_models(args.output_dir)

    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"\nSentis-compatible models exported to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Copy *.sentis.onnx files to Unity Assets/Models/")
    print("  2. Implement ISTFT in C# (n_fft=16, hop_length=4)")
    print("  3. Implement Qwen2 tokenizer in C#")
    print("  4. Implement KV cache management in C#")


if __name__ == '__main__':
    main()
