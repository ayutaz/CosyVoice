#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
HiFT ONNX Verification Script

This script verifies the exported ONNX models by comparing their outputs
with the PyTorch model outputs.

Usage:
    python scripts/verify_hift_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
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
import onnxruntime as ort

from hyperpyyaml import load_hyperpyyaml
from scipy.io import wavfile


def load_hift_model(model_dir: str, device: str = 'cpu'):
    """Load HiFT model from model directory"""

    yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')

    print(f"Loading config from {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})

    hift = configs['hift']

    hift_path = os.path.join(model_dir, 'hift.pt')
    if os.path.exists(hift_path):
        print(f"Loading weights from {hift_path}")
        state_dict = torch.load(hift_path, map_location=device)
        hift.load_state_dict(state_dict)

    hift.eval()
    hift.to(device)

    return hift


def verify_f0_predictor(
    hift: torch.nn.Module,
    onnx_path: str,
    mel_len: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """Verify F0 Predictor ONNX against PyTorch"""

    print(f"\n{'='*60}")
    print("Verifying F0 Predictor")
    print(f"{'='*60}")

    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Generate test input
    mel = torch.randn(1, 80, mel_len)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = hift.f0_predictor(mel, finalize=True)

    # ONNX inference
    onnx_inputs = {'mel': mel.numpy()}
    onnx_output = session.run(None, onnx_inputs)[0]

    # Compare
    pytorch_np = pytorch_output.numpy()
    max_diff = np.abs(pytorch_np - onnx_output).max()
    mean_diff = np.abs(pytorch_np - onnx_output).mean()

    print(f"PyTorch output shape: {pytorch_np.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # Check if close
    is_close = np.allclose(pytorch_np, onnx_output, rtol=rtol, atol=atol)
    print(f"Outputs match (rtol={rtol}, atol={atol}): {is_close}")

    if not is_close:
        # More lenient check
        is_close_lenient = np.allclose(pytorch_np, onnx_output, rtol=1e-2, atol=1e-3)
        print(f"Outputs match (lenient rtol=1e-2, atol=1e-3): {is_close_lenient}")

    return is_close


def verify_hift_decoder(
    hift: torch.nn.Module,
    onnx_path: str,
    mel_len: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """Verify HiFT Decoder ONNX against PyTorch"""

    print(f"\n{'='*60}")
    print("Verifying HiFT Decoder")
    print(f"{'='*60}")

    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Generate test input
    mel = torch.randn(1, 80, mel_len)

    # Generate source STFT using PyTorch model
    with torch.no_grad():
        # Get F0
        f0 = hift.f0_predictor(mel, finalize=True)

        # Upsample F0
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)

        # Source generation
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)

        # STFT of source
        s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
        source_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

    print(f"Source STFT shape: {source_stft.shape}")

    # Import wrapper for PyTorch comparison
    from scripts.export_hift_onnx import HiFTDecoderWrapper

    wrapper = HiFTDecoderWrapper(hift)
    wrapper.eval()

    # PyTorch inference
    with torch.no_grad():
        pytorch_mag, pytorch_phase = wrapper(mel, source_stft)

    # ONNX inference
    onnx_inputs = {
        'mel': mel.numpy(),
        'source_stft': source_stft.numpy()
    }
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_mag = onnx_outputs[0]
    onnx_phase = onnx_outputs[1]

    # Compare magnitude
    pytorch_mag_np = pytorch_mag.numpy()
    max_diff_mag = np.abs(pytorch_mag_np - onnx_mag).max()
    mean_diff_mag = np.abs(pytorch_mag_np - onnx_mag).mean()

    print(f"\nMagnitude comparison:")
    print(f"  PyTorch shape: {pytorch_mag_np.shape}")
    print(f"  ONNX shape: {onnx_mag.shape}")
    print(f"  Max absolute difference: {max_diff_mag:.6e}")
    print(f"  Mean absolute difference: {mean_diff_mag:.6e}")

    # Compare phase
    pytorch_phase_np = pytorch_phase.numpy()
    max_diff_phase = np.abs(pytorch_phase_np - onnx_phase).max()
    mean_diff_phase = np.abs(pytorch_phase_np - onnx_phase).mean()

    print(f"\nPhase comparison:")
    print(f"  PyTorch shape: {pytorch_phase_np.shape}")
    print(f"  ONNX shape: {onnx_phase.shape}")
    print(f"  Max absolute difference: {max_diff_phase:.6e}")
    print(f"  Mean absolute difference: {mean_diff_phase:.6e}")

    # Check if close
    mag_close = np.allclose(pytorch_mag_np, onnx_mag, rtol=rtol, atol=atol)
    phase_close = np.allclose(pytorch_phase_np, onnx_phase, rtol=rtol, atol=atol)

    print(f"\nMagnitude match (rtol={rtol}, atol={atol}): {mag_close}")
    print(f"Phase match (rtol={rtol}, atol={atol}): {phase_close}")

    return mag_close and phase_close


def test_full_pipeline(
    hift: torch.nn.Module,
    f0_onnx_path: str,
    decoder_onnx_path: str,
    mel_len: int = 100,
    output_wav: str = None
):
    """Test the full pipeline with ONNX models and compare with PyTorch"""

    print(f"\n{'='*60}")
    print("Testing Full Pipeline (ONNX + C# equivalents)")
    print(f"{'='*60}")

    # Create ONNX sessions
    f0_session = ort.InferenceSession(f0_onnx_path, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(decoder_onnx_path, providers=['CPUExecutionProvider'])

    # Generate test mel spectrogram
    mel = torch.randn(1, 80, mel_len)
    mel_np = mel.numpy()

    print(f"Input mel shape: {mel.shape}")

    # Step 1: F0 Prediction (ONNX)
    f0_onnx = f0_session.run(None, {'mel': mel_np})[0]
    print(f"F0 (ONNX) shape: {f0_onnx.shape}")
    print(f"F0 range: [{f0_onnx.min():.2f}, {f0_onnx.max():.2f}]")

    # Step 2: Source generation (PyTorch - would be C# in Unity)
    with torch.no_grad():
        f0_torch = torch.from_numpy(f0_onnx)
        s = hift.f0_upsamp(f0_torch[:, None]).transpose(1, 2)
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)

        # STFT
        s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
        source_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

    source_stft_np = source_stft.numpy()
    print(f"Source STFT shape: {source_stft_np.shape}")

    # Step 3: Decoder (ONNX)
    decoder_outputs = decoder_session.run(None, {
        'mel': mel_np,
        'source_stft': source_stft_np
    })
    magnitude = decoder_outputs[0]
    phase = decoder_outputs[1]

    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Phase shape: {phase.shape}")

    # Step 4: ISTFT (PyTorch - would be C# in Unity)
    magnitude_torch = torch.from_numpy(magnitude)
    phase_torch = torch.from_numpy(phase)

    with torch.no_grad():
        audio = hift._istft(magnitude_torch, phase_torch)
        audio = torch.clamp(audio, -hift.audio_limit, hift.audio_limit)

    audio_np = audio.numpy()
    print(f"Audio shape: {audio_np.shape}")
    print(f"Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")

    # Compare with PyTorch full pipeline
    with torch.no_grad():
        pytorch_audio, _ = hift.inference(mel.transpose(1, 2).transpose(1, 2))

    pytorch_audio_np = pytorch_audio.numpy()
    max_diff = np.abs(pytorch_audio_np - audio_np).max()
    mean_diff = np.abs(pytorch_audio_np - audio_np).mean()

    print(f"\nComparison with PyTorch full pipeline:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Save audio if requested
    if output_wav:
        sample_rate = hift.sampling_rate
        audio_int16 = (audio_np[0] * 32767).astype(np.int16)
        wavfile.write(output_wav, sample_rate, audio_int16)
        print(f"\nSaved audio to {output_wav}")

    return audio_np


def main():
    parser = argparse.ArgumentParser(description='Verify HiFT ONNX exports')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='pretrained_models/Fun-CosyVoice3-0.5B',
        help='Path to model directory'
    )
    parser.add_argument(
        '--onnx_dir',
        type=str,
        default=None,
        help='ONNX directory (default: model_dir/onnx)'
    )
    parser.add_argument(
        '--mel_len',
        type=int,
        default=100,
        help='Mel spectrogram length for testing'
    )
    parser.add_argument(
        '--output_wav',
        type=str,
        default=None,
        help='Output WAV file path (optional)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Test FP16 models'
    )

    args = parser.parse_args()

    if args.onnx_dir is None:
        args.onnx_dir = os.path.join(args.model_dir, 'onnx')

    # Determine model suffix
    suffix = '_fp16' if args.fp16 else '_fp32'

    f0_path = os.path.join(args.onnx_dir, f'hift_f0_predictor{suffix}.onnx')
    decoder_path = os.path.join(args.onnx_dir, f'hift_decoder{suffix}.onnx')

    # Check files exist
    for path in [f0_path, decoder_path]:
        if not os.path.exists(path):
            print(f"Error: ONNX file not found: {path}")
            print("Please run export_hift_onnx.py first.")
            return

    # Load PyTorch model
    hift = load_hift_model(args.model_dir)

    # Verify F0 Predictor
    f0_ok = verify_f0_predictor(hift, f0_path, args.mel_len)

    # Verify Decoder
    decoder_ok = verify_hift_decoder(hift, decoder_path, args.mel_len)

    # Test full pipeline
    test_full_pipeline(
        hift,
        f0_path,
        decoder_path,
        args.mel_len,
        args.output_wav
    )

    # Summary
    print(f"\n{'='*60}")
    print("Verification Summary")
    print(f"{'='*60}")
    print(f"F0 Predictor: {'PASS' if f0_ok else 'FAIL'}")
    print(f"Decoder: {'PASS' if decoder_ok else 'FAIL'}")

    if f0_ok and decoder_ok:
        print("\nAll verifications passed!")
    else:
        print("\nSome verifications failed. Check tolerances or model export.")


if __name__ == '__main__':
    main()
