#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Flow Decoder ONNX Export Script

This script exports the Flow Matching decoder components to ONNX format for Unity Sentis.

Export Strategy:
- flow.decoder.estimator: Already exported by official script (DiT model, ~1.3GB FP32)
- input_embedding: Embedding layer for speech tokens
- spk_embed_affine_layer: Linear projection for speaker embedding
- pre_lookahead_layer: Convolutional layer for lookahead processing
- Euler Solver: Implemented in C# (calls estimator multiple times)

Usage:
    python scripts/export_flow_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root and third_party to path BEFORE other imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

import numpy as np
import torch
import torch.nn as nn

from hyperpyyaml import load_hyperpyyaml


class TokenEmbeddingWrapper(nn.Module):
    """Wrapper for token embedding export"""

    def __init__(self, flow):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.vocab_size = flow.vocab_size

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token: [batch, seq_len] speech token indices
        Returns:
            embedded: [batch, seq_len, embed_dim] embedded tokens
        """
        # Clamp to valid range
        token = torch.clamp(token, min=0, max=self.vocab_size - 1)
        return self.input_embedding(token)


class SpeakerEmbedProjectionWrapper(nn.Module):
    """Wrapper for speaker embedding projection"""

    def __init__(self, flow):
        super().__init__()
        self.spk_embed_affine_layer = flow.spk_embed_affine_layer

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [batch, spk_embed_dim] speaker embedding (normalized)
        Returns:
            projected: [batch, output_size] projected embedding
        """
        # Note: F.normalize should be done before calling this
        return self.spk_embed_affine_layer(embedding)


class PreLookaheadWrapper(nn.Module):
    """Wrapper for pre-lookahead layer"""

    def __init__(self, flow):
        super().__init__()
        self.pre_lookahead_layer = flow.pre_lookahead_layer
        self.token_mel_ratio = flow.token_mel_ratio

    def forward(self, token_embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embedded: [batch, seq_len, embed_dim] embedded tokens
        Returns:
            h: [batch, seq_len * token_mel_ratio, output_size] processed features
        """
        h = self.pre_lookahead_layer(token_embedded)
        # Repeat interleave for token_mel_ratio
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        return h


def load_flow_model(model_dir: str, device: str = 'cpu'):
    """Load Flow model from model directory"""

    yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found in {model_dir}")

    print(f"Loading config from {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'hift': None})

    flow = configs['flow']

    # Load weights
    flow_path = os.path.join(model_dir, 'flow.pt')
    if os.path.exists(flow_path):
        print(f"Loading weights from {flow_path}")
        state_dict = torch.load(flow_path, map_location=device)
        flow.load_state_dict(state_dict)
    else:
        print(f"Warning: {flow_path} not found, using random weights")

    flow.eval()
    flow.to(device)

    return flow


def export_token_embedding(
    flow: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export token embedding to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting Token Embedding to ONNX")
    print(f"{'='*60}")

    wrapper = TokenEmbeddingWrapper(flow)
    wrapper.eval()

    # Dummy input
    batch_size = 1
    seq_len = 100
    token = torch.randint(0, flow.vocab_size, (batch_size, seq_len))

    if fp16:
        wrapper = wrapper.half()

    print(f"Input shape: token {token.shape}")
    print(f"Vocab size: {flow.vocab_size}")

    with torch.no_grad():
        output = wrapper(token)
    print(f"Output shape: embedded {output.shape}")

    torch.onnx.export(
        wrapper,
        (token,),
        output_path,
        opset_version=opset_version,
        input_names=['token'],
        output_names=['embedded'],
        dynamic_axes={
            'token': {0: 'batch', 1: 'seq_len'},
            'embedded': {0: 'batch', 1: 'seq_len'}
        },
        do_constant_folding=True,
    )

    print(f"Exported to {output_path}")

    # Verify
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model validation passed!")

    return output_path


def export_speaker_projection(
    flow: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export speaker embedding projection to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting Speaker Embedding Projection to ONNX")
    print(f"{'='*60}")

    wrapper = SpeakerEmbedProjectionWrapper(flow)
    wrapper.eval()

    # Dummy input (192 is spk_embed_dim from CAMPPlus)
    batch_size = 1
    spk_embed_dim = 192
    embedding = torch.randn(batch_size, spk_embed_dim)

    if fp16:
        wrapper = wrapper.half()
        embedding = embedding.half()

    print(f"Input shape: embedding {embedding.shape}")

    with torch.no_grad():
        output = wrapper(embedding)
    print(f"Output shape: projected {output.shape}")

    torch.onnx.export(
        wrapper,
        (embedding,),
        output_path,
        opset_version=opset_version,
        input_names=['embedding'],
        output_names=['projected'],
        dynamic_axes={
            'embedding': {0: 'batch'},
            'projected': {0: 'batch'}
        },
        do_constant_folding=True,
    )

    print(f"Exported to {output_path}")

    # Verify
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model validation passed!")

    return output_path


def export_pre_lookahead(
    flow: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export pre-lookahead layer to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting Pre-Lookahead Layer to ONNX")
    print(f"{'='*60}")

    wrapper = PreLookaheadWrapper(flow)
    wrapper.eval()

    # Dummy input
    batch_size = 1
    seq_len = 100
    input_size = flow.input_size  # Should be 896 for CosyVoice3
    token_embedded = torch.randn(batch_size, seq_len, input_size)

    if fp16:
        wrapper = wrapper.half()
        token_embedded = token_embedded.half()

    print(f"Input shape: token_embedded {token_embedded.shape}")
    print(f"Token mel ratio: {flow.token_mel_ratio}")

    with torch.no_grad():
        output = wrapper(token_embedded)
    print(f"Output shape: h {output.shape}")

    torch.onnx.export(
        wrapper,
        (token_embedded,),
        output_path,
        opset_version=opset_version,
        input_names=['token_embedded'],
        output_names=['h'],
        dynamic_axes={
            'token_embedded': {0: 'batch', 1: 'seq_len'},
            'h': {0: 'batch', 1: 'mel_len'}
        },
        do_constant_folding=True,
    )

    print(f"Exported to {output_path}")

    # Verify
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model validation passed!")

    return output_path


def convert_estimator_to_fp16(
    model_dir: str,
    output_dir: str
):
    """Convert the existing FP32 estimator to FP16"""

    print(f"\n{'='*60}")
    print("Converting Flow Decoder Estimator to FP16")
    print(f"{'='*60}")

    input_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
    output_path = os.path.join(output_dir, 'flow_decoder_estimator_fp16.onnx')

    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found")
        print("Run the official export_onnx.py first:")
        print(f"  python cosyvoice/bin/export_onnx.py --model_dir {model_dir}")
        return None

    try:
        from onnxruntime.transformers import float16

        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        float16.convert_float_to_float16(
            input_path,
            output_path,
            keep_io_types=True,
            op_block_list=['LayerNormalization', 'ReduceMean']
        )

        print(f"Conversion complete!")

        # Check sizes
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"FP32 size: {input_size:.1f} MB")
        print(f"FP16 size: {output_size:.1f} MB")
        print(f"Reduction: {(1 - output_size/input_size) * 100:.1f}%")

        return output_path

    except ImportError:
        print("Warning: onnxruntime.transformers not available")
        print("Install with: uv add onnxruntime-transformers")
        return None


def verify_existing_estimator(model_dir: str):
    """Verify the existing flow decoder estimator ONNX"""

    print(f"\n{'='*60}")
    print("Verifying Existing Flow Decoder Estimator")
    print(f"{'='*60}")

    onnx_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')

    if not os.path.exists(onnx_path):
        print(f"Warning: {onnx_path} not found")
        return False

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Get input/output info
    print("\nInput tensors:")
    for input in session.get_inputs():
        print(f"  {input.name}: {input.shape} ({input.type})")

    print("\nOutput tensors:")
    for output in session.get_outputs():
        print(f"  {output.name}: {output.shape} ({output.type})")

    # Test inference (batch_size=2 as expected by the model)
    batch_size = 2
    seq_len = 100
    out_channels = 80

    x = np.random.randn(batch_size, out_channels, seq_len).astype(np.float32)
    mask = np.ones((batch_size, 1, seq_len), dtype=np.float32)
    mu = np.random.randn(batch_size, out_channels, seq_len).astype(np.float32)
    t = np.random.rand(batch_size).astype(np.float32)
    spks = np.random.randn(batch_size, out_channels).astype(np.float32)
    cond = np.random.randn(batch_size, out_channels, seq_len).astype(np.float32)

    ort_inputs = {
        'x': x,
        'mask': mask,
        'mu': mu,
        't': t,
        'spks': spks,
        'cond': cond
    }

    output = session.run(None, ort_inputs)[0]
    print(f"\nTest inference:")
    print(f"  Input x shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    return True


def main():
    parser = argparse.ArgumentParser(description='Export Flow components to ONNX')
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
        help='Output directory (default: model_dir/onnx)'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=15,
        help='ONNX opset version (default: 15 for Unity Sentis)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Export in FP16 precision'
    )
    parser.add_argument(
        '--convert_estimator',
        action='store_true',
        help='Convert existing estimator to FP16'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'onnx')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Opset version: {args.opset_version}")
    print(f"FP16: {args.fp16}")

    # Verify existing estimator
    verify_existing_estimator(args.model_dir)

    # Load model
    flow = load_flow_model(args.model_dir)

    # Print model info
    print(f"\nFlow Configuration:")
    print(f"  - Input size: {flow.input_size}")
    print(f"  - Output size: {flow.output_size}")
    print(f"  - Vocab size: {flow.vocab_size}")
    print(f"  - Token mel ratio: {flow.token_mel_ratio}")
    print(f"  - Pre-lookahead len: {flow.pre_lookahead_len}")

    # Export components
    precision_suffix = '_fp16' if args.fp16 else '_fp32'

    # 1. Token Embedding
    embed_path = os.path.join(args.output_dir, f'flow_token_embedding{precision_suffix}.onnx')
    export_token_embedding(flow, embed_path, args.opset_version, args.fp16)

    # 2. Speaker Projection
    spk_path = os.path.join(args.output_dir, f'flow_speaker_projection{precision_suffix}.onnx')
    export_speaker_projection(flow, spk_path, args.opset_version, args.fp16)

    # 3. Pre-Lookahead Layer
    lookahead_path = os.path.join(args.output_dir, f'flow_pre_lookahead{precision_suffix}.onnx')
    export_pre_lookahead(flow, lookahead_path, args.opset_version, args.fp16)

    # 4. Optionally convert estimator to FP16
    if args.convert_estimator:
        convert_estimator_to_fp16(args.model_dir, args.output_dir)

    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"Token Embedding: {embed_path}")
    print(f"Speaker Projection: {spk_path}")
    print(f"Pre-Lookahead: {lookahead_path}")
    print(f"\nExisting (from official export):")
    print(f"  flow.decoder.estimator.fp32.onnx (~1.3GB)")

    print(f"\n{'='*60}")
    print("Unity Implementation Notes")
    print(f"{'='*60}")
    print("""
For Unity Sentis, implement the Flow inference as follows:

1. Normalize speaker embedding: embedding = F.normalize(embedding)
2. Project speaker embedding: spks = SpeakerProjection(embedding)
3. Embed tokens: token_embed = TokenEmbedding(speech_tokens)
4. Apply mask: token_embed = token_embed * mask
5. Pre-lookahead: h = PreLookahead(token_embed)
6. Prepare conditions: cond with prompt_feat
7. Run Euler solver (10 steps):
   for t in [0.0, 0.1, ..., 0.9]:
       velocity = Estimator(x, mask, mu=h, t, spks, cond)
       x = x + velocity * dt
8. Output mel spectrogram: mel = x[:, :, prompt_len:]
""")


if __name__ == '__main__':
    main()
