#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
HiFT (HiFi-GAN with iSTFT) ONNX Export Script

This script exports the HiFT vocoder components to ONNX format for Unity Sentis.

Export Strategy:
- F0 Predictor: mel -> F0 (separate ONNX)
- Decoder Core: mel + source_stft -> magnitude/phase (separate ONNX)
- STFT/ISTFT: Implemented in C# (n_fft=16 is trivial)
- Source Generation: Implemented in C# (sine wave from F0)

Usage:
    python scripts/export_hift_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
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


class F0PredictorWrapper(nn.Module):
    """Wrapper for F0 Predictor ONNX export"""

    def __init__(self, f0_predictor: nn.Module):
        super().__init__()
        self.f0_predictor = f0_predictor

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch, 80, seq_len] mel spectrogram
        Returns:
            f0: [batch, seq_len] fundamental frequency
        """
        # Call the predictor with finalize=True for non-streaming
        return self.f0_predictor(mel, finalize=True)


class HiFTDecoderWrapper(nn.Module):
    """
    Wrapper for HiFT Decoder ONNX export.

    This exports the core neural network part, excluding STFT/ISTFT operations.
    Input: mel spectrogram + source STFT
    Output: magnitude and phase (before ISTFT)
    """

    def __init__(self, hift: nn.Module):
        super().__init__()
        self.hift = hift
        # Copy necessary attributes
        self.num_upsamples = hift.num_upsamples
        self.num_kernels = hift.num_kernels
        self.lrelu_slope = hift.lrelu_slope
        self.istft_params = hift.istft_params

        # Copy layers
        self.conv_pre = hift.conv_pre
        self.ups = hift.ups
        self.source_downs = hift.source_downs
        self.source_resblocks = hift.source_resblocks
        self.resblocks = hift.resblocks
        self.conv_post = hift.conv_post
        self.reflection_pad = hift.reflection_pad

    def forward(
        self,
        mel: torch.Tensor,
        source_stft: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [batch, 80, mel_len] mel spectrogram
            source_stft: [batch, n_fft+2, stft_len] concatenated STFT real and imag
                         where n_fft=16, so shape is [batch, 18, stft_len]
        Returns:
            magnitude: [batch, n_fft//2+1, out_len] = [batch, 9, out_len]
            phase: [batch, n_fft//2+1, out_len] = [batch, 9, out_len]
        """
        x = self.conv_pre(mel)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Fusion with source
            si = self.source_downs[i](source_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            # ResBlocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude and phase
        n_fft_half = self.istft_params["n_fft"] // 2 + 1  # 9
        magnitude = torch.exp(x[:, :n_fft_half, :])
        phase = torch.sin(x[:, n_fft_half:, :])

        return magnitude, phase


def load_hift_model(model_dir: str, device: str = 'cpu') -> nn.Module:
    """Load HiFT model from model directory"""

    yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found in {model_dir}")

    print(f"Loading config from {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})

    hift = configs['hift']

    # Load weights
    hift_path = os.path.join(model_dir, 'hift.pt')
    if os.path.exists(hift_path):
        print(f"Loading weights from {hift_path}")
        state_dict = torch.load(hift_path, map_location=device)
        hift.load_state_dict(state_dict)
    else:
        print(f"Warning: {hift_path} not found, using random weights")

    hift.eval()
    hift.to(device)

    return hift


def export_f0_predictor(
    hift: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export F0 Predictor to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting F0 Predictor to ONNX")
    print(f"{'='*60}")

    wrapper = F0PredictorWrapper(hift.f0_predictor)
    wrapper.eval()

    # Dummy input
    batch_size = 1
    mel_len = 100
    mel = torch.randn(batch_size, 80, mel_len)

    if fp16:
        wrapper = wrapper.half()
        mel = mel.half()

    # Export
    print(f"Input shape: mel {mel.shape}")

    with torch.no_grad():
        output = wrapper(mel)
    print(f"Output shape: f0 {output.shape}")

    torch.onnx.export(
        wrapper,
        (mel,),
        output_path,
        opset_version=opset_version,
        input_names=['mel'],
        output_names=['f0'],
        dynamic_axes={
            'mel': {0: 'batch', 2: 'mel_len'},
            'f0': {0: 'batch', 1: 'mel_len'}
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


def compute_source_stft_length(hift: nn.Module, mel_len: int) -> int:
    """
    Compute the required source_stft length for a given mel length.
    This accounts for causal convolution padding and upsampling.
    """
    # Run a dummy forward to get the actual sizes
    with torch.no_grad():
        mel = torch.randn(1, 80, mel_len)

        # Get F0
        f0 = hift.f0_predictor(mel, finalize=True)

        # Upsample F0
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)

        # Source generation
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)

        # STFT of source
        s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
        stft_len = s_stft_real.shape[2]

        return stft_len


def export_hift_decoder(
    hift: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export HiFT Decoder to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting HiFT Decoder to ONNX")
    print(f"{'='*60}")

    wrapper = HiFTDecoderWrapper(hift)
    wrapper.eval()

    batch_size = 1
    mel_len = 100
    n_fft = hift.istft_params["n_fft"]  # 16

    # Compute the actual STFT length by running the model
    stft_len = compute_source_stft_length(hift, mel_len)
    print(f"Computed source_stft length for mel_len={mel_len}: {stft_len}")

    mel = torch.randn(batch_size, 80, mel_len)
    source_stft = torch.randn(batch_size, n_fft + 2, stft_len)

    if fp16:
        wrapper = wrapper.half()
        mel = mel.half()
        source_stft = source_stft.half()

    print(f"Input shapes: mel {mel.shape}, source_stft {source_stft.shape}")

    with torch.no_grad():
        magnitude, phase = wrapper(mel, source_stft)
    print(f"Output shapes: magnitude {magnitude.shape}, phase {phase.shape}")

    torch.onnx.export(
        wrapper,
        (mel, source_stft),
        output_path,
        opset_version=opset_version,
        input_names=['mel', 'source_stft'],
        output_names=['magnitude', 'phase'],
        dynamic_axes={
            'mel': {0: 'batch', 2: 'mel_len'},
            'source_stft': {0: 'batch', 2: 'stft_len'},
            'magnitude': {0: 'batch', 2: 'out_len'},
            'phase': {0: 'batch', 2: 'out_len'}
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


def export_full_hift_for_reference(
    hift: nn.Module,
    output_dir: str,
    opset_version: int = 15,
):
    """
    Export a reference version that includes everything.
    This may not work in Unity due to STFT/ISTFT, but useful for Python verification.
    """

    print(f"\n{'='*60}")
    print("Exporting Full HiFT (for reference/verification)")
    print(f"{'='*60}")

    class FullHiFTWrapper(nn.Module):
        def __init__(self, hift):
            super().__init__()
            self.hift = hift

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            """
            Args:
                mel: [batch, mel_len, 80] mel spectrogram (transposed for easier use)
            Returns:
                audio: [batch, audio_len] waveform
            """
            # Transpose to [batch, 80, mel_len]
            mel = mel.transpose(1, 2)
            audio, _ = self.hift.inference(mel)
            return audio

    wrapper = FullHiFTWrapper(hift)
    wrapper.eval()

    batch_size = 1
    mel_len = 100
    mel = torch.randn(batch_size, mel_len, 80)

    output_path = os.path.join(output_dir, 'hift_full_reference.onnx')

    try:
        with torch.no_grad():
            audio = wrapper(mel)
        print(f"Input shape: mel {mel.shape}")
        print(f"Output shape: audio {audio.shape}")

        torch.onnx.export(
            wrapper,
            (mel,),
            output_path,
            opset_version=opset_version,
            input_names=['mel'],
            output_names=['audio'],
            dynamic_axes={
                'mel': {0: 'batch', 1: 'mel_len'},
                'audio': {0: 'batch', 1: 'audio_len'}
            },
            do_constant_folding=True,
        )
        print(f"Exported to {output_path}")

    except Exception as e:
        print(f"Warning: Full export failed (expected due to STFT): {e}")
        print("This is expected - use the split exports for Unity.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export HiFT to ONNX')
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
        '--device',
        type=str,
        default='cpu',
        help='Device for export (cpu recommended)'
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'onnx')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Opset version: {args.opset_version}")
    print(f"FP16: {args.fp16}")

    # Load model
    hift = load_hift_model(args.model_dir, args.device)

    # Print model info
    print(f"\nHiFT Configuration:")
    print(f"  - Sample rate: {hift.sampling_rate}")
    print(f"  - ISTFT params: n_fft={hift.istft_params['n_fft']}, hop_len={hift.istft_params['hop_len']}")
    print(f"  - Num upsamples: {hift.num_upsamples}")
    print(f"  - Num kernels: {hift.num_kernels}")

    # Export components
    precision_suffix = '_fp16' if args.fp16 else '_fp32'

    # 1. Export F0 Predictor
    f0_path = os.path.join(args.output_dir, f'hift_f0_predictor{precision_suffix}.onnx')
    export_f0_predictor(hift, f0_path, args.opset_version, args.fp16)

    # 2. Export Decoder
    decoder_path = os.path.join(args.output_dir, f'hift_decoder{precision_suffix}.onnx')
    export_hift_decoder(hift, decoder_path, args.opset_version, args.fp16)

    # 3. Try full export (for reference)
    export_full_hift_for_reference(hift, args.output_dir, args.opset_version)

    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"F0 Predictor: {f0_path}")
    print(f"Decoder: {decoder_path}")
    print(f"\nFor Unity Sentis, you need to implement in C#:")
    print("  1. Source generation (F0 -> sine waves)")
    print("  2. STFT (source -> source_stft)")
    print("  3. ISTFT (magnitude/phase -> audio)")
    print(f"\nISTFT parameters: n_fft=16, hop_len=4")


if __name__ == '__main__':
    main()
