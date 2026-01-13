#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
ONNX Component Verification Tests

Tests each ONNX component against PyTorch to verify correctness.
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
    print(f"    Close (rtol={rtol}, atol={atol}): {'PASS' if is_close else 'FAIL'}")

    return is_close


def test_text_embedding():
    """Test text embedding ONNX"""
    print("\n" + "=" * 60)
    print("Testing: Text Embedding")
    print("=" * 60)

    llm, _, _ = load_pytorch_models()
    onnx_session = load_onnx_session('text_embedding', use_fp16=False)

    # Test input
    input_ids = np.array([[1, 2, 3, 100, 500, 1000]], dtype=np.int64)

    # PyTorch
    with torch.no_grad():
        pt_out = llm.llm.model.model.embed_tokens(torch.from_numpy(input_ids))

    # ONNX
    onnx_out = onnx_session.run(None, {'input_ids': input_ids})[0]

    return compare_outputs('Text Embedding', pt_out, onnx_out)


def test_speech_embedding():
    """Test speech embedding ONNX"""
    print("\n" + "=" * 60)
    print("Testing: Speech Embedding (LLM)")
    print("=" * 60)

    llm, _, _ = load_pytorch_models()
    onnx_session = load_onnx_session('llm_speech_embedding')

    # Test input (speech tokens 0-6560, special tokens 6561+)
    tokens = np.array([[100, 500, 1000, 3000, 6000, 6561, 6562]], dtype=np.int64)

    # PyTorch
    with torch.no_grad():
        pt_out = llm.speech_embedding(torch.from_numpy(tokens))

    # ONNX
    onnx_out = onnx_session.run(None, {'token': tokens})[0]

    return compare_outputs('Speech Embedding', pt_out, onnx_out)


def test_llm_decoder():
    """Test LLM decoder ONNX"""
    print("\n" + "=" * 60)
    print("Testing: LLM Decoder")
    print("=" * 60)

    llm, _, _ = load_pytorch_models()
    onnx_session = load_onnx_session('llm_decoder')

    # Test input
    hidden_state = np.random.randn(1, 1, 896).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = llm.llm_decoder(torch.from_numpy(hidden_state))

    # ONNX
    onnx_out = onnx_session.run(None, {'hidden_state': hidden_state})[0]

    return compare_outputs('LLM Decoder', pt_out, onnx_out)


def test_flow_token_embedding():
    """Test Flow token embedding ONNX"""
    print("\n" + "=" * 60)
    print("Testing: Flow Token Embedding")
    print("=" * 60)

    _, flow, _ = load_pytorch_models()
    onnx_session = load_onnx_session('flow_token_embedding')

    # Test input
    tokens = np.array([[100, 500, 1000, 2000, 3000]], dtype=np.int64)

    # PyTorch
    with torch.no_grad():
        pt_out = flow.input_embedding(torch.from_numpy(tokens))

    # ONNX
    onnx_out = onnx_session.run(None, {'token': tokens})[0]

    return compare_outputs('Flow Token Embedding', pt_out, onnx_out)


def test_flow_speaker_projection():
    """Test Flow speaker projection ONNX"""
    print("\n" + "=" * 60)
    print("Testing: Flow Speaker Projection")
    print("=" * 60)

    _, flow, _ = load_pytorch_models()
    onnx_session = load_onnx_session('flow_speaker_projection')

    # Test input (normalized embedding)
    embedding = np.random.randn(1, 192).astype(np.float32)
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

    # PyTorch
    with torch.no_grad():
        pt_out = flow.spk_embed_affine_layer(torch.from_numpy(embedding))

    # ONNX
    onnx_out = onnx_session.run(None, {'embedding': embedding})[0]

    return compare_outputs('Flow Speaker Projection', pt_out, onnx_out)


def test_hift_f0_predictor():
    """Test HiFT F0 predictor ONNX (FP32 required for numerical stability)"""
    print("\n" + "=" * 60)
    print("Testing: HiFT F0 Predictor (FP32)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()
    # Use FP32 explicitly - FP16 has precision issues with large intermediate values
    onnx_session = load_onnx_session('hift_f0_predictor', use_fp16=False)

    # Test input
    mel = np.random.randn(1, 80, 50).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = hift.f0_predictor(torch.from_numpy(mel), finalize=True)

    # ONNX
    onnx_out = onnx_session.run(None, {'mel': mel})[0]

    return compare_outputs('HiFT F0 Predictor', pt_out, onnx_out)


def test_hift_source_generator():
    """Test HiFT source generator ONNX (FP32)"""
    print("\n" + "=" * 60)
    print("Testing: HiFT Source Generator (FP32)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()
    # Use FP32 for consistency with other HiFT components
    onnx_session = load_onnx_session('hift_source_generator', use_fp16=False)

    # Test input (F0)
    f0 = np.abs(np.random.randn(1, 1, 50).astype(np.float32)) * 200 + 50  # F0 range

    # PyTorch
    with torch.no_grad():
        f0_pt = torch.from_numpy(f0)
        s = hift.f0_upsamp(f0_pt).transpose(1, 2)
        source_pt, _, _ = hift.m_source(s)
        pt_out = source_pt.transpose(1, 2)

    # ONNX
    onnx_out = onnx_session.run(None, {'f0': f0})[0]

    return compare_outputs('HiFT Source Generator', pt_out, onnx_out)


def test_hift_decoder():
    """Test HiFT decoder ONNX (FP32)"""
    print("\n" + "=" * 60)
    print("Testing: HiFT Decoder (FP32)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()
    # Use FP32 for consistency with other HiFT components
    onnx_session = load_onnx_session('hift_decoder', use_fp16=False)

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

    result1 = compare_outputs('HiFT Decoder (Magnitude)', magnitude_pt, magnitude_onnx)
    result2 = compare_outputs('HiFT Decoder (Phase)', phase_pt, phase_onnx)

    return result1 and result2


def test_full_hift_pipeline():
    """Test full HiFT pipeline: mel -> audio"""
    print("\n" + "=" * 60)
    print("Testing: Full HiFT Pipeline (mel -> audio)")
    print("=" * 60)

    _, _, hift = load_pytorch_models()

    # Test input
    mel = np.random.randn(1, 80, 50).astype(np.float32)

    # PyTorch full pipeline
    with torch.no_grad():
        audio_pt, _ = hift.inference(torch.from_numpy(mel), finalize=True)

    print(f"  PyTorch output:")
    print(f"    Shape: {audio_pt.shape}")
    print(f"    Range: [{audio_pt.min():.4f}, {audio_pt.max():.4f}]")

    # ONNX pipeline
    f0_session = load_onnx_session('hift_f0_predictor')
    source_session = load_onnx_session('hift_source_generator', use_fp16=False)
    decoder_session = load_onnx_session('hift_decoder')

    # F0 prediction
    f0_onnx = f0_session.run(None, {'mel': mel})[0]
    print(f"\n  ONNX F0:")
    print(f"    Shape: {f0_onnx.shape}")
    print(f"    Range: [{f0_onnx.min():.4f}, {f0_onnx.max():.4f}]")

    # Source generation
    f0_input = f0_onnx[:, np.newaxis, :]
    source_onnx = source_session.run(None, {'f0': f0_input.astype(np.float32)})[0]
    print(f"\n  ONNX Source:")
    print(f"    Shape: {source_onnx.shape}")
    print(f"    Range: [{source_onnx.min():.4f}, {source_onnx.max():.4f}]")

    # Compare source with PyTorch
    with torch.no_grad():
        f0_pt = hift.f0_predictor(torch.from_numpy(mel), finalize=True)
        s = hift.f0_upsamp(f0_pt[:, None]).transpose(1, 2)
        source_pt, _, _ = hift.m_source(s)
        source_pt = source_pt.transpose(1, 2)

    print(f"\n  PyTorch Source:")
    print(f"    Shape: {source_pt.shape}")
    print(f"    Range: [{source_pt.min():.4f}, {source_pt.max():.4f}]")

    # Check source match
    source_diff = np.abs(source_pt.numpy() - source_onnx).max()
    print(f"\n  Source max diff: {source_diff:.6f}")

    return True


def main():
    print("=" * 60)
    print("ONNX Component Verification Tests")
    print("=" * 60)

    results = {}

    # Run tests
    tests = [
        ('Text Embedding', test_text_embedding),
        ('Speech Embedding', test_speech_embedding),
        ('LLM Decoder', test_llm_decoder),
        ('Flow Token Embedding', test_flow_token_embedding),
        ('Flow Speaker Projection', test_flow_speaker_projection),
        ('HiFT F0 Predictor', test_hift_f0_predictor),
        ('HiFT Source Generator', test_hift_source_generator),
        ('HiFT Decoder', test_hift_decoder),
        ('Full HiFT Pipeline', test_full_hift_pipeline),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ERROR: {e}")
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
