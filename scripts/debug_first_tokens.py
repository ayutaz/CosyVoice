#!/usr/bin/env python3
"""
Debug first token generation issue in Pure ONNX inference.

Investigates why there's an "a~" sound at the beginning of generated audio.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
ONNX_DIR = os.path.join(MODEL_DIR, 'onnx')


def load_onnx_models():
    """Load ONNX models"""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ['CPUExecutionProvider']

    models = {
        'text_embedding': ort.InferenceSession(
            os.path.join(ONNX_DIR, 'text_embedding_fp32.onnx'), so, providers=providers
        ),
        'speech_embedding': ort.InferenceSession(
            os.path.join(ONNX_DIR, 'llm_speech_embedding_fp16.onnx'), so, providers=providers
        ),
        'backbone_initial': ort.InferenceSession(
            os.path.join(ONNX_DIR, 'llm_backbone_initial_fp16.onnx'), so, providers=providers
        ),
        'backbone_decode': ort.InferenceSession(
            os.path.join(ONNX_DIR, 'llm_backbone_decode_fp16.onnx'), so, providers=providers
        ),
        'llm_decoder': ort.InferenceSession(
            os.path.join(ONNX_DIR, 'llm_decoder_fp16.onnx'), so, providers=providers
        ),
    }
    return models


def load_pytorch_model():
    """Load PyTorch model"""
    from cosyvoice.cli.cosyvoice import CosyVoice3
    return CosyVoice3(MODEL_DIR)


def compare_first_tokens():
    """Compare first token generation between ONNX and PyTorch"""
    print("=" * 60)
    print("First Token Generation Comparison")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    onnx_models = load_onnx_models()
    cosyvoice = load_pytorch_model()

    # Load tokenizer
    from transformers import AutoTokenizer
    qwen_path = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    # Test text
    test_text = "<|en|>Hello world."
    text_tokens = tokenizer.encode(test_text, add_special_tokens=False)
    text_token_ids = np.array([text_tokens], dtype=np.int64)

    print(f"\nTest text: {test_text}")
    print(f"Text tokens: {text_tokens}")
    print(f"Text token count: {len(text_tokens)}")

    # Special tokens
    speech_token_size = 6561
    sos_token = speech_token_size  # 6561
    eos_token = speech_token_size + 1  # 6562
    task_id_token = speech_token_size + 2  # 6563

    print(f"\nSpecial tokens:")
    print(f"  SOS: {sos_token}")
    print(f"  EOS: {eos_token}")
    print(f"  TASK_ID: {task_id_token}")

    # ==================== ONNX ====================
    print("\n" + "=" * 60)
    print("ONNX First Token Generation")
    print("=" * 60)

    # Get embeddings
    text_emb_onnx = onnx_models['text_embedding'].run(None, {'input_ids': text_token_ids})[0]
    sos_emb_onnx = onnx_models['speech_embedding'].run(None, {'token': np.array([[sos_token]], dtype=np.int64)})[0]
    task_id_emb_onnx = onnx_models['speech_embedding'].run(None, {'token': np.array([[task_id_token]], dtype=np.int64)})[0]

    print(f"\nEmbedding shapes:")
    print(f"  SOS: {sos_emb_onnx.shape}")
    print(f"  Text: {text_emb_onnx.shape}")
    print(f"  TASK_ID: {task_id_emb_onnx.shape}")

    # Build LLM input
    lm_input_onnx = np.concatenate([sos_emb_onnx, text_emb_onnx, task_id_emb_onnx], axis=1).astype(np.float32)
    print(f"  LLM input: {lm_input_onnx.shape}")

    # Initial forward pass
    seq_len = lm_input_onnx.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.float32)

    initial_outputs = onnx_models['backbone_initial'].run(
        None, {'inputs_embeds': lm_input_onnx, 'attention_mask': attention_mask}
    )
    hidden_states_onnx = initial_outputs[0]
    past_kv_onnx = initial_outputs[1]

    print(f"\nInitial pass outputs:")
    print(f"  Hidden states: {hidden_states_onnx.shape}")
    print(f"  KV cache: {past_kv_onnx.shape}")

    # Get first logits
    logits_onnx = onnx_models['llm_decoder'].run(
        None, {'hidden_state': hidden_states_onnx[:, -1:, :]}
    )[0]

    print(f"\nFirst logits:")
    print(f"  Shape: {logits_onnx.shape}")
    print(f"  Range: [{logits_onnx.min():.4f}, {logits_onnx.max():.4f}]")

    # Get top-5 tokens
    logits_flat = logits_onnx.flatten()[:speech_token_size]
    top5_idx = np.argsort(logits_flat)[-5:][::-1]
    top5_probs = np.exp(logits_flat[top5_idx]) / np.exp(logits_flat).sum()

    print(f"\nONNX Top-5 first tokens:")
    for idx, prob in zip(top5_idx, top5_probs):
        print(f"  Token {idx}: prob={prob:.4f}")

    # Generate first 10 tokens with greedy sampling
    print("\nONNX First 10 tokens (greedy):")
    onnx_tokens = []
    current_hidden = hidden_states_onnx[:, -1:, :]
    current_kv = past_kv_onnx

    for i in range(10):
        logits = onnx_models['llm_decoder'].run(None, {'hidden_state': current_hidden})[0]
        token = np.argmax(logits.flatten()[:speech_token_size])
        onnx_tokens.append(token)
        print(f"  Step {i}: token={token}")

        if token == eos_token:
            break

        # Get next embedding
        next_emb = onnx_models['speech_embedding'].run(
            None, {'token': np.array([[token]], dtype=np.int64)}
        )[0]

        # Decode step
        total_len = seq_len + len(onnx_tokens)
        attn_mask = np.ones((1, total_len), dtype=np.float32)

        decode_outputs = onnx_models['backbone_decode'].run(
            None, {
                'inputs_embeds': next_emb.astype(np.float32),
                'attention_mask': attn_mask,
                'past_key_values': current_kv
            }
        )
        current_hidden = decode_outputs[0]
        current_kv = decode_outputs[1]

    # ==================== PyTorch ====================
    print("\n" + "=" * 60)
    print("PyTorch First Token Generation")
    print("=" * 60)

    with torch.no_grad():
        # Get device from model
        device = next(cosyvoice.model.llm.llm.parameters()).device
        print(f"\nPyTorch model device: {device}")

        # Get embeddings
        text_token_tensor = torch.tensor([text_tokens], dtype=torch.long, device=device)
        text_emb_pt = cosyvoice.model.llm.llm.model.model.embed_tokens(text_token_tensor)

        sos_emb_pt = cosyvoice.model.llm.speech_embedding.weight[sos_token].reshape(1, 1, -1)
        task_id_emb_pt = cosyvoice.model.llm.speech_embedding.weight[task_id_token].reshape(1, 1, -1)

        print(f"\nEmbedding shapes:")
        print(f"  SOS: {sos_emb_pt.shape}")
        print(f"  Text: {text_emb_pt.shape}")
        print(f"  TASK_ID: {task_id_emb_pt.shape}")

        # Build LLM input
        lm_input_pt = torch.concat([sos_emb_pt, text_emb_pt, task_id_emb_pt], dim=1)
        print(f"  LLM input: {lm_input_pt.shape}")

        # Compare inputs
        input_diff = np.abs(lm_input_pt.cpu().numpy() - lm_input_onnx).max()
        print(f"\n  LLM input max diff (ONNX vs PT): {input_diff:.6f}")

        # Initial forward pass
        # Note: cosyvoice.model.llm.llm is Qwen2Encoder, .model is Qwen2ForCausalLM, .model is Qwen2Model
        qwen2_model = cosyvoice.model.llm.llm.model.model  # Get the actual Qwen2Model

        # Create attention mask
        seq_len_pt = lm_input_pt.shape[1]
        attention_mask_pt = torch.ones((1, seq_len_pt), device=device)

        outputs_pt = qwen2_model(
            inputs_embeds=lm_input_pt,
            attention_mask=attention_mask_pt,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        hidden_states_pt = outputs_pt.hidden_states[-1]
        past_kv_pt = outputs_pt.past_key_values

        print(f"\nInitial pass outputs:")
        print(f"  Hidden states: {hidden_states_pt.shape}")
        print(f"  KV cache layers: {len(past_kv_pt)}")

        # Compare hidden states
        hidden_pt_np = hidden_states_pt.cpu().numpy()
        hidden_diff = np.abs(hidden_pt_np - hidden_states_onnx).max()
        hidden_mean_diff = np.abs(hidden_pt_np - hidden_states_onnx).mean()
        print(f"  Hidden states max diff: {hidden_diff:.6f}")
        print(f"  Hidden states mean diff: {hidden_mean_diff:.6f}")

        # Per-position hidden state diff
        print(f"\n  Per-position hidden state max diff:")
        for pos in range(min(5, hidden_pt_np.shape[1])):
            pos_diff = np.abs(hidden_pt_np[0, pos, :] - hidden_states_onnx[0, pos, :]).max()
            print(f"    Position {pos}: {pos_diff:.6f}")
        if hidden_pt_np.shape[1] > 5:
            last_pos_diff = np.abs(hidden_pt_np[0, -1, :] - hidden_states_onnx[0, -1, :]).max()
            print(f"    Position {hidden_pt_np.shape[1]-1} (last): {last_pos_diff:.6f}")

        # Get first logits
        logits_pt = cosyvoice.model.llm.llm_decoder(hidden_states_pt[:, -1, :])

        print(f"\nFirst logits:")
        print(f"  Shape: {logits_pt.shape}")
        print(f"  Range: [{logits_pt.min():.4f}, {logits_pt.max():.4f}]")

        # Compare logits
        logits_pt_np = logits_pt.cpu().numpy()
        logits_diff = np.abs(logits_pt_np - logits_onnx.squeeze()).max()
        logits_mean_diff = np.abs(logits_pt_np - logits_onnx.squeeze()).mean()
        print(f"  Logits max diff: {logits_diff:.6f}")
        print(f"  Logits mean diff: {logits_mean_diff:.6f}")

        # Check if top tokens are the same
        pt_top10 = np.argsort(logits_pt_np.flatten()[:speech_token_size])[-10:][::-1]
        onnx_top10 = np.argsort(logits_onnx.flatten()[:speech_token_size])[-10:][::-1]
        print(f"\n  PT Top-10 tokens: {pt_top10.tolist()}")
        print(f"  ONNX Top-10 tokens: {onnx_top10.tolist()}")

        # Get top-5 tokens
        logits_flat_pt = logits_pt.flatten()[:speech_token_size]
        top5_idx_pt = torch.argsort(logits_flat_pt, descending=True)[:5]
        top5_probs_pt = torch.softmax(logits_flat_pt[top5_idx_pt], dim=-1)

        print(f"\nPyTorch Top-5 first tokens:")
        for idx, prob in zip(top5_idx_pt.tolist(), top5_probs_pt.tolist()):
            print(f"  Token {idx}: prob={prob:.4f}")

        # Generate first 10 tokens with greedy sampling
        print("\nPyTorch First 10 tokens (greedy):")
        pt_tokens = []
        current_hidden_pt = hidden_states_pt[:, -1:, :]
        current_kv_pt = past_kv_pt

        for i in range(10):
            logits = cosyvoice.model.llm.llm_decoder(current_hidden_pt.squeeze(1))
            token = torch.argmax(logits[:, :speech_token_size], dim=-1).item()
            pt_tokens.append(token)
            print(f"  Step {i}: token={token}")

            if token == eos_token:
                break

            # Get next embedding
            next_emb = cosyvoice.model.llm.speech_embedding(torch.tensor([[token]], device=device))

            # Decode step - update attention mask for total length
            total_len_pt = seq_len_pt + len(pt_tokens)
            attn_mask_pt = torch.ones((1, total_len_pt), device=device)

            outputs = qwen2_model(
                inputs_embeds=next_emb,
                attention_mask=attn_mask_pt,
                past_key_values=current_kv_pt,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True
            )
            current_hidden_pt = outputs.hidden_states[-1]
            current_kv_pt = outputs.past_key_values

    # ==================== Comparison ====================
    print("\n" + "=" * 60)
    print("Token Comparison")
    print("=" * 60)

    print(f"\nONNX tokens:    {onnx_tokens}")
    print(f"PyTorch tokens: {pt_tokens}")

    matches = sum(1 for a, b in zip(onnx_tokens, pt_tokens) if a == b)
    print(f"\nMatching: {matches}/{min(len(onnx_tokens), len(pt_tokens))}")

    if onnx_tokens[0] != pt_tokens[0]:
        print(f"\nWARNING: First token differs!")
        print(f"  ONNX first token: {onnx_tokens[0]}")
        print(f"  PyTorch first token: {pt_tokens[0]}")


if __name__ == '__main__':
    compare_first_tokens()
