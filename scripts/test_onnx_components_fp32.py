#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
ONNX Component Verification Tests (FP32 for HiFT)
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


def load_pytorch_models():
    """Load PyTorch models"""
    yaml_path = os.path.join(MODEL_DIR, 'cosyvoice3.yaml')
    with open(yaml_path, 'r') as f:
        qwen_path = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path})

    # Load LLM
    llm = configs['llm']
    llm_state = torch.load(os.path.join(MODEL_DIR, 'llm.pt'), map_location='cpu')
    llm.load_state_dict(llm_state)
    llm.eval()

    # Load Flow
    flow = configs['flow']
    flow_state = torch.load(os.path.join(MODEL_DIR, 'flow.pt'), map_location='cpu')
    flow.load_state_dict(flow_state)
    flow.eval()

    # Load HiFT
    hift = configs['hift']
    hift_state = {k.replace('generator.', ''): v
                  for k, v in torch.load(os.path.join(MODEL_DIR, 'hift.pt'), map_location='cpu').items()}
    hift.load_state_dict(hift_state)
    hift.eval()

    return llm, flow, hift


def load_onnx_session(name, use_fp16=True):
    """Load ONNX session"""
    suffix = '_fp16' if use_fp16 else '_fp32'
    so = ort.SessionOptions()
    so.log_severity_level = 3

    # Try different naming patterns
    paths = [
        os.path.join(ONNX_DIR, f'{name}{suffix}.onnx'),
        os.path.join(ONNX_DIR, f'{name}_fp32.onnx'),
        os.path.join(ONNX_DIR, f'{name}.onnx'),
    ]

    for path in paths:
        if os.path.exists(path):
            return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    raise FileNotFoundError(f"ONNX model not found: {name}")


def compare_outputs(name, pt_output, onnx_output, rtol=1e-2, atol=1e-2):
    """Compare PyTorch and ONNX outputs"""
    if isinstance(pt_output, torch.Tensor):
        pt_output = pt_output.detach().numpy()

    # Reshape if needed
    if pt_output.shape != onnx_output.shape:
        if pt_output.squeeze().shape == onnx_output.squeeze().shape:
            pt_output = pt_output.squeeze()
            onnx_output = onnx_output.squeeze()
            if len(pt_output.shape) == 1:
                pt_output = pt_output.reshape(1, -1)
                onnx_output = onnx_output.reshape(1, -1)

    # Calculate differences
    diff = np.abs(pt_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = np.mean(diff / (np.abs(pt_output) + 1e-8))

    # Check if close
    is_close = np.allclose(pt_output, onnx_output, rtol=rtol, atol=atol)

    print(f"  {name}:")
    print(f"    Shape: PT={pt_output.shape}, ONNX={onnx_output.shape}")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Mean diff: {mean_diff:.6f}")
    print(f"    Relative diff: {rel_diff:.6f}")
    status = "PASS" if is_close else "FAIL"
    print(f"    Close (rtol={rtol}, atol={atol}): {status}")

    return is_close


def test_hift_f0_predictor_fp32():
    """Test HiFT F0 predictor ONNX (FP32)"""
    print("\n" + "=" * 60)
    print("Testing: HiFT F0 Predictor (FP32)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()

    # Load FP32 model specifically
    so = ort.SessionOptions()
    so.log_severity_level = 3
    onnx_session = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'hift_f0_predictor_fp32.onnx'),
        so, providers=['CPUExecutionProvider']
    )

    # Test input
    mel = np.random.randn(1, 80, 50).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = hift.f0_predictor(torch.from_numpy(mel), finalize=True)

    # ONNX
    onnx_out = onnx_session.run(None, {'mel': mel})[0]

    return compare_outputs('HiFT F0 Predictor (FP32)', pt_out, onnx_out)


def test_hift_source_generator_fp32():
    """Test HiFT source generator ONNX (FP32)"""
    print("\n" + "=" * 60)
    print("Testing: HiFT Source Generator (FP32)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    onnx_session = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'hift_source_generator_fp32.onnx'),
        so, providers=['CPUExecutionProvider']
    )

    # Test input (F0)
    f0 = np.abs(np.random.randn(1, 1, 50).astype(np.float32)) * 200 + 50

    # PyTorch
    with torch.no_grad():
        f0_pt = torch.from_numpy(f0)
        s = hift.f0_upsamp(f0_pt).transpose(1, 2)
        source_pt, _, _ = hift.m_source(s)
        pt_out = source_pt.transpose(1, 2)

    # ONNX
    onnx_out = onnx_session.run(None, {'f0': f0})[0]

    return compare_outputs('HiFT Source Generator (FP32)', pt_out, onnx_out)


def test_hift_decoder_fp32():
    """Test HiFT decoder ONNX (FP32)"""
    print("\n" + "=" * 60)
    print("Testing: HiFT Decoder (FP32)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    onnx_session = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'hift_decoder_fp32.onnx'),
        so, providers=['CPUExecutionProvider']
    )

    # Test input
    mel = np.random.randn(1, 80, 50).astype(np.float32)

    # First get source from PyTorch
    with torch.no_grad():
        f0 = hift.f0_predictor(torch.from_numpy(mel), finalize=True)
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = hift.m_source(s)
        source = source.transpose(1, 2)

        # Compute STFT of source
        s_stft_real, s_stft_imag = hift._stft(source.squeeze(1))
        source_stft_pt = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # Run decoder
        x = hift.conv_pre(torch.from_numpy(mel))
        for i in range(hift.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, hift.lrelu_slope)
            x = hift.ups[i](x)
            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)
            si = hift.source_downs[i](source_stft_pt)
            si = hift.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(hift.num_kernels):
                if xs is None:
                    xs = hift.resblocks[i * hift.num_kernels + j](x)
                else:
                    xs += hift.resblocks[i * hift.num_kernels + j](x)
            x = xs / hift.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = hift.conv_post(x)

        # Output is magnitude and phase
        magnitude_pt = torch.exp(x[:, :hift.istft_params["n_fft"] // 2 + 1, :])
        phase_pt = torch.sin(x[:, hift.istft_params["n_fft"] // 2 + 1:, :])

    # ONNX
    source_stft = source_stft_pt.numpy().astype(np.float32)
    onnx_out = onnx_session.run(None, {'mel': mel, 'source_stft': source_stft})
    magnitude_onnx = onnx_out[0]
    phase_onnx = onnx_out[1]

    result1 = compare_outputs('HiFT Decoder Magnitude (FP32)', magnitude_pt, magnitude_onnx)
    result2 = compare_outputs('HiFT Decoder Phase (FP32)', phase_pt, phase_onnx)

    return result1 and result2


def test_full_hift_pipeline_fp32():
    """Test full HiFT pipeline with FP32 ONNX models"""
    import torch
    from scipy.signal import get_window

    print("\n" + "=" * 60)
    print("Testing: Full HiFT Pipeline (FP32 ONNX)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()

    so = ort.SessionOptions()
    so.log_severity_level = 3

    f0_session = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'hift_f0_predictor_fp32.onnx'),
        so, providers=['CPUExecutionProvider']
    )
    source_session = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'hift_source_generator_fp32.onnx'),
        so, providers=['CPUExecutionProvider']
    )
    decoder_session = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'hift_decoder_fp32.onnx'),
        so, providers=['CPUExecutionProvider']
    )

    # Test input
    mel = np.random.randn(1, 80, 50).astype(np.float32)

    # PyTorch full pipeline
    with torch.no_grad():
        audio_pt, _ = hift.inference(torch.from_numpy(mel), finalize=True)

    print(f"  PyTorch audio:")
    print(f"    Shape: {audio_pt.shape}")
    print(f"    Range: [{audio_pt.min():.4f}, {audio_pt.max():.4f}]")

    # ONNX pipeline
    # 1. F0 prediction
    f0_onnx = f0_session.run(None, {'mel': mel})[0]
    print(f"\n  ONNX F0: shape={f0_onnx.shape}, range=[{f0_onnx.min():.4f}, {f0_onnx.max():.4f}]")

    # 2. Source generation
    f0_input = f0_onnx[:, np.newaxis, :].astype(np.float32)
    source_onnx = source_session.run(None, {'f0': f0_input})[0]
    print(f"  ONNX Source: shape={source_onnx.shape}, range=[{source_onnx.min():.4f}, {source_onnx.max():.4f}]")

    # 3. Compute source STFT (using torch for exact HiFT compatibility)
    from scipy.signal import get_window
    import torch

    n_fft = 16
    hop_len = 4
    source_1d = source_onnx[0, 0, :]

    # Create Hann window matching PyTorch HiFT
    window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))

    # Compute STFT using torch (same as HiFT._stft)
    source_t = torch.from_numpy(source_1d.astype(np.float32)).unsqueeze(0)
    spec = torch.stft(source_t, n_fft, hop_len, n_fft, window=window, return_complex=True)
    spec = torch.view_as_real(spec)  # [B, F, T, 2]
    s_stft_real = spec[..., 0].numpy()
    s_stft_imag = spec[..., 1].numpy()
    source_stft = np.concatenate([s_stft_real, s_stft_imag], axis=1).astype(np.float32)
    print(f"  Source STFT: shape={source_stft.shape}")

    # 4. Decoder
    onnx_out = decoder_session.run(None, {'mel': mel, 'source_stft': source_stft})
    magnitude_onnx = onnx_out[0]
    phase_onnx = onnx_out[1]
    print(f"  Decoder output - magnitude: {magnitude_onnx.shape}, phase: {phase_onnx.shape}")

    # 5. ISTFT using torch (same as HiFT._istft)
    magnitude_onnx = np.clip(magnitude_onnx, None, 100.0)
    real = magnitude_onnx * np.cos(phase_onnx)
    imag = magnitude_onnx * np.sin(phase_onnx)

    # Convert to torch and compute ISTFT
    real_t = torch.from_numpy(real[0].astype(np.float32))
    imag_t = torch.from_numpy(imag[0].astype(np.float32))
    complex_spec = torch.complex(real_t, imag_t)
    audio_onnx_t = torch.istft(complex_spec, n_fft, hop_len, n_fft, window=window)
    audio_onnx = audio_onnx_t.numpy()[np.newaxis, :]

    print(f"\n  ONNX audio:")
    print(f"    Shape: {audio_onnx.shape}")
    print(f"    Range: [{audio_onnx.min():.4f}, {audio_onnx.max():.4f}]")

    # Compare with PyTorch
    audio_pt_np = audio_pt.numpy()

    # Truncate to same length
    min_len = min(audio_pt_np.shape[1], audio_onnx.shape[1])
    audio_pt_np = audio_pt_np[:, :min_len]
    audio_onnx = audio_onnx[:, :min_len]

    diff = np.abs(audio_pt_np - audio_onnx)
    print(f"\n  Audio comparison:")
    print(f"    Max diff: {diff.max():.6f}")
    print(f"    Mean diff: {diff.mean():.6f}")

    # Correlation
    corr = np.corrcoef(audio_pt_np.flatten(), audio_onnx.flatten())[0, 1]
    print(f"    Correlation: {corr:.6f}")

    return corr > 0.9


def main():
    print("=" * 60)
    print("ONNX Component Verification Tests (FP32 for HiFT)")
    print("=" * 60)

    results = {}

    # Run tests
    tests = [
        ('HiFT F0 Predictor (FP32)', test_hift_f0_predictor_fp32),
        ('HiFT Source Generator (FP32)', test_hift_source_generator_fp32),
        ('HiFT Decoder (FP32)', test_hift_decoder_fp32),
        ('Full HiFT Pipeline (FP32)', test_full_hift_pipeline_fp32),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = 'PASS' if result else 'FAIL'
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
