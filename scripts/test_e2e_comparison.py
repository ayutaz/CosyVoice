#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
End-to-End Comparison Test: Pure ONNX vs PyTorch

This test compares the full pipeline outputs at each stage to identify
where the ONNX inference diverges from PyTorch.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
ONNX_DIR = os.path.join(MODEL_DIR, 'onnx')


def load_cosyvoice():
    """Load CosyVoice model"""
    from cosyvoice.cli.cosyvoice import CosyVoice3
    return CosyVoice3(MODEL_DIR)


def test_llm_output_comparison():
    """Compare LLM speech token generation between ONNX and PyTorch"""
    print("\n" + "=" * 60)
    print("Test: LLM Speech Token Comparison")
    print("=" * 60)

    # Load PyTorch model
    cosyvoice = load_cosyvoice()

    # Test text
    test_text = "<|en|>Hello world."

    # PyTorch LLM inference
    print("\n1. PyTorch LLM Inference:")
    with torch.no_grad():
        # Tokenize
        text_token, text_token_len = cosyvoice.frontend.text_normalize_v3(
            test_text, split=False, text_frontend=False
        )

        # Get embeddings
        text_token = text_token.to(cosyvoice.model.device)
        text_emb = cosyvoice.model.llm.llm.model.model.embed_tokens(text_token)

        # Get special token embeddings
        speech_token_size = 6561
        sos_emb = cosyvoice.model.llm.speech_embedding.weight[speech_token_size].reshape(1, 1, -1)
        task_id_emb = cosyvoice.model.llm.speech_embedding.weight[speech_token_size + 2].reshape(1, 1, -1)

        # Build LLM input
        lm_input = torch.concat([sos_emb, text_emb, task_id_emb], dim=1)

        print(f"  Text tokens: {text_token.shape}")
        print(f"  LLM input shape: {lm_input.shape}")

        # Run LLM sampling (limited tokens for comparison)
        speech_tokens_pt = []
        max_tokens = 50

        # Initial forward pass
        outputs = cosyvoice.model.llm.llm(inputs_embeds=lm_input, use_cache=True)
        hidden_states = outputs.last_hidden_state[:, -1:, :]
        past_key_values = outputs.past_key_values

        for i in range(max_tokens):
            # Get logits
            logits = cosyvoice.model.llm.llm_decoder(hidden_states)

            # Sample (greedy for reproducibility)
            token = torch.argmax(logits[:, -1, :speech_token_size], dim=-1)
            speech_tokens_pt.append(token.item())

            # Check EOS
            if token.item() == speech_token_size + 1:
                break

            # Get next embedding
            next_emb = cosyvoice.model.llm.speech_embedding(token.unsqueeze(0))

            # Next forward
            outputs = cosyvoice.model.llm.llm(
                inputs_embeds=next_emb,
                past_key_values=past_key_values,
                use_cache=True
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        print(f"  PyTorch generated {len(speech_tokens_pt)} tokens")
        print(f"  First 10 tokens: {speech_tokens_pt[:10]}")
        print(f"  Token range: [{min(speech_tokens_pt)}, {max(speech_tokens_pt)}]")

    # Now compare with ONNX
    print("\n2. ONNX LLM Inference:")

    # Load ONNX models
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ['CPUExecutionProvider']

    text_embedding = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'text_embedding_fp32.onnx'),
        so, providers=providers
    )
    llm_backbone_initial = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'llm_backbone_initial_fp16.onnx'),
        so, providers=providers
    )
    llm_backbone_decode = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'llm_backbone_decode_fp16.onnx'),
        so, providers=providers
    )
    llm_decoder = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'llm_decoder_fp16.onnx'),
        so, providers=providers
    )
    speech_embedding = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'llm_speech_embedding_fp16.onnx'),
        so, providers=providers
    )

    # Tokenize (same as PyTorch)
    from transformers import AutoTokenizer
    qwen_path = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    text_tokens = tokenizer.encode(test_text, add_special_tokens=False)
    text_token_ids = np.array([text_tokens], dtype=np.int64)

    # Get embeddings
    text_emb_onnx = text_embedding.run(None, {'input_ids': text_token_ids})[0]

    # Get special token embeddings
    sos_token = np.array([[speech_token_size]], dtype=np.int64)
    task_id_token = np.array([[speech_token_size + 2]], dtype=np.int64)
    sos_emb_onnx = speech_embedding.run(None, {'token': sos_token})[0]
    task_id_emb_onnx = speech_embedding.run(None, {'token': task_id_token})[0]

    # Build LLM input
    lm_input_onnx = np.concatenate([sos_emb_onnx, text_emb_onnx, task_id_emb_onnx], axis=1)

    print(f"  Text tokens: {text_token_ids.shape}")
    print(f"  LLM input shape: {lm_input_onnx.shape}")

    # Compare LLM inputs
    lm_input_pt_np = lm_input.cpu().numpy()
    input_diff = np.abs(lm_input_pt_np - lm_input_onnx).max()
    print(f"  LLM input max diff: {input_diff:.6f}")

    # Run ONNX LLM
    speech_tokens_onnx = []

    # Initial forward
    initial_outputs = llm_backbone_initial.run(None, {
        'input_embeds': lm_input_onnx.astype(np.float32)
    })
    hidden_state = initial_outputs[0][:, -1:, :]
    kv_cache = initial_outputs[1:]

    for i in range(max_tokens):
        # Get logits
        logits = llm_decoder.run(None, {'hidden_state': hidden_state.astype(np.float32)})[0]

        # Sample (greedy)
        token = np.argmax(logits[..., :speech_token_size], axis=-1).item()
        speech_tokens_onnx.append(token)

        # Check EOS
        if token == speech_token_size + 1:
            break

        # Get next embedding
        token_input = np.array([[token]], dtype=np.int64)
        next_emb = speech_embedding.run(None, {'token': token_input})[0]

        # Prepare KV cache inputs
        decode_inputs = {'input_embeds': next_emb.astype(np.float32)}
        for j, kv in enumerate(kv_cache):
            decode_inputs[f'past_key_values.{j}'] = kv.astype(np.float32)

        # Next forward
        decode_outputs = llm_backbone_decode.run(None, decode_inputs)
        hidden_state = decode_outputs[0]
        kv_cache = decode_outputs[1:]

    print(f"  ONNX generated {len(speech_tokens_onnx)} tokens")
    print(f"  First 10 tokens: {speech_tokens_onnx[:10]}")
    print(f"  Token range: [{min(speech_tokens_onnx)}, {max(speech_tokens_onnx)}]")

    # Compare tokens
    print("\n3. Token Comparison:")
    min_len = min(len(speech_tokens_pt), len(speech_tokens_onnx))
    matches = sum(1 for a, b in zip(speech_tokens_pt[:min_len], speech_tokens_onnx[:min_len]) if a == b)
    print(f"  Matching tokens: {matches}/{min_len} ({100*matches/min_len:.1f}%)")

    if speech_tokens_pt != speech_tokens_onnx:
        print("  WARNING: Token sequences differ!")
        # Find first difference
        for i, (pt, onnx) in enumerate(zip(speech_tokens_pt, speech_tokens_onnx)):
            if pt != onnx:
                print(f"  First difference at position {i}: PT={pt}, ONNX={onnx}")
                break

    return speech_tokens_pt, speech_tokens_onnx


def test_flow_comparison(speech_tokens):
    """Compare Flow mel generation"""
    print("\n" + "=" * 60)
    print("Test: Flow Mel Generation Comparison")
    print("=" * 60)

    # Use same speech tokens for both
    tokens = np.array([speech_tokens[:50]], dtype=np.int64)  # Limit for speed

    # Load PyTorch
    cosyvoice = load_cosyvoice()

    # PyTorch Flow
    print("\n1. PyTorch Flow:")
    with torch.no_grad():
        token_tensor = torch.from_numpy(tokens).to(cosyvoice.model.device)

        # Get embedding (no prompt)
        embedding = torch.randn(1, 192).to(cosyvoice.model.device)

        # Flow inference (simplified - just check token embedding)
        token_emb_pt = cosyvoice.model.flow.input_embedding(token_tensor)
        print(f"  Token embedding shape: {token_emb_pt.shape}")
        print(f"  Token embedding range: [{token_emb_pt.min():.4f}, {token_emb_pt.max():.4f}]")

    # ONNX Flow
    print("\n2. ONNX Flow:")
    so = ort.SessionOptions()
    so.log_severity_level = 3

    flow_token_emb = ort.InferenceSession(
        os.path.join(ONNX_DIR, 'flow_token_embedding_fp16.onnx'),
        so, providers=['CPUExecutionProvider']
    )

    token_emb_onnx = flow_token_emb.run(None, {'token': tokens})[0]
    print(f"  Token embedding shape: {token_emb_onnx.shape}")
    print(f"  Token embedding range: [{token_emb_onnx.min():.4f}, {token_emb_onnx.max():.4f}]")

    # Compare
    diff = np.abs(token_emb_pt.cpu().numpy() - token_emb_onnx).max()
    print(f"\n  Max difference: {diff:.6f}")

    return diff < 0.01


def test_speech_token_validity(tokens):
    """Check if speech tokens are in valid range"""
    print("\n" + "=" * 60)
    print("Test: Speech Token Validity")
    print("=" * 60)

    speech_token_size = 6561

    print(f"  Total tokens: {len(tokens)}")
    print(f"  Valid range: [0, {speech_token_size - 1}]")

    invalid_tokens = [t for t in tokens if t < 0 or t >= speech_token_size]
    print(f"  Invalid tokens: {len(invalid_tokens)}")

    if invalid_tokens:
        print(f"  Invalid token values: {invalid_tokens[:10]}...")

    # Check token distribution
    unique_tokens = len(set(tokens))
    print(f"  Unique tokens: {unique_tokens}")

    # Check for repeated patterns (sign of generation issues)
    if len(tokens) > 10:
        # Check last 10 tokens for repetition
        last_10 = tokens[-10:]
        if len(set(last_10)) <= 2:
            print("  WARNING: Repetitive pattern detected in last tokens!")

    return len(invalid_tokens) == 0


def main():
    print("=" * 60)
    print("End-to-End ONNX vs PyTorch Comparison")
    print("=" * 60)

    # Test 1: LLM comparison
    pt_tokens, onnx_tokens = test_llm_output_comparison()

    # Test 2: Token validity
    print("\nPyTorch tokens validity:")
    test_speech_token_validity(pt_tokens)

    print("\nONNX tokens validity:")
    test_speech_token_validity(onnx_tokens)

    # Test 3: Flow comparison (using PyTorch tokens)
    test_flow_comparison(pt_tokens)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PyTorch generated: {len(pt_tokens)} tokens")
    print(f"ONNX generated: {len(onnx_tokens)} tokens")

    if len(pt_tokens) != len(onnx_tokens):
        print("WARNING: Different number of tokens generated!")

    token_match_rate = sum(1 for a, b in zip(pt_tokens, onnx_tokens) if a == b) / min(len(pt_tokens), len(onnx_tokens))
    print(f"Token match rate: {100*token_match_rate:.1f}%")


if __name__ == '__main__':
    main()
