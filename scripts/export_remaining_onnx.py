#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Export remaining components to ONNX for Unity Sentis compatibility.

Components to export:
1. Text embedding (Qwen2 embed_tokens)
2. HiFT source generation (F0 upsample + source module)
3. Vocabulary file for C# tokenizer

Usage:
    python scripts/export_remaining_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import onnx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party' / 'Matcha-TTS'))

from hyperpyyaml import load_hyperpyyaml


class TextEmbeddingWrapper(nn.Module):
    """Wrapper for Qwen2 text embedding layer"""

    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class SourceGeneratorWrapper(nn.Module):
    """Wrapper for HiFT source generation (F0 -> source signal)"""

    def __init__(self, f0_upsamp, m_source):
        super().__init__()
        self.f0_upsamp = f0_upsamp
        self.m_source = m_source

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        # f0: [batch, 1, time]
        s = self.f0_upsamp(f0).transpose(1, 2)  # [batch, time_up, 1]
        source, _, _ = self.m_source(s)
        return source.transpose(1, 2)  # [batch, 1, time_up]


def export_text_embedding(model_dir: str, output_dir: str, use_fp16: bool = False):
    """Export Qwen2 text embedding layer"""
    print("\n" + "=" * 60)
    print("Exporting Text Embedding (Qwen2 embed_tokens)")
    print("=" * 60)

    # Load config
    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path})

    # Load LLM
    llm_config = configs['llm']
    llm_path = os.path.join(model_dir, 'llm.pt')
    print(f"Loading LLM from {llm_path}...")
    state_dict = torch.load(llm_path, map_location='cpu')
    llm_config.load_state_dict(state_dict)
    llm_config.eval()

    # Get embed_tokens
    embed_tokens = llm_config.llm.model.model.embed_tokens
    wrapper = TextEmbeddingWrapper(embed_tokens)
    wrapper.eval()

    # Export
    dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    output_path = os.path.join(output_dir, 'text_embedding_fp32.onnx')

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'embeddings': {0: 'batch', 1: 'seq_len'}
        },
        opset_version=17
    )

    # Verify
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print(f"  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Get embedding info
    vocab_size, embed_dim = embed_tokens.weight.shape
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embed_dim}")


def export_source_generator(model_dir: str, output_dir: str, use_fp16: bool = False):
    """Export HiFT source generator (F0 -> source signal)"""
    print("\n" + "=" * 60)
    print("Exporting HiFT Source Generator")
    print("=" * 60)

    # Load config
    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path})

    # Load HiFT
    hift_config = configs['hift']
    hift_path = os.path.join(model_dir, 'hift.pt')
    print(f"Loading HiFT from {hift_path}...")
    hift_state_dict = {k.replace('generator.', ''): v
                       for k, v in torch.load(hift_path, map_location='cpu').items()}
    hift_config.load_state_dict(hift_state_dict)
    hift_config.eval()

    # Create wrapper
    wrapper = SourceGeneratorWrapper(
        hift_config.f0_upsamp,
        hift_config.m_source
    )
    wrapper.eval()

    # Export
    # F0 has shape [batch, 1, time]
    dummy_f0 = torch.randn(1, 1, 100)
    output_path = os.path.join(output_dir, 'hift_source_generator_fp32.onnx')

    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(
            wrapper,
            dummy_f0,
            output_path,
            input_names=['f0'],
            output_names=['source'],
            dynamic_axes={
                'f0': {0: 'batch', 2: 'time'},
                'source': {0: 'batch', 2: 'time_up'}
            },
            opset_version=17
        )

        # Verify
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"  Saved: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"  Failed to export source generator: {e}")
        print("  This component may require custom ONNX ops")


def export_vocabulary(model_dir: str, output_dir: str):
    """Export Qwen2 vocabulary for C# tokenizer"""
    print("\n" + "=" * 60)
    print("Exporting Vocabulary")
    print("=" * 60)

    from transformers import AutoTokenizer

    # Load tokenizer
    qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
    if os.path.exists(qwen_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)

    # Export vocabulary
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, 'qwen2_vocab.json')

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {vocab_path}")
    print(f"  Vocab size: {len(vocab)}")

    # Export special tokens
    special_tokens = {
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
    }

    special_path = os.path.join(output_dir, 'qwen2_special_tokens.json')
    with open(special_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {special_path}")

    # Export merges for BPE (if available)
    try:
        if hasattr(tokenizer, 'bpe_ranks') or hasattr(tokenizer, 'merges'):
            print("  Note: BPE merges file should be copied from tokenizer directory")
    except:
        pass


def export_speech_embedding_extended(model_dir: str, output_dir: str, use_fp16: bool = False):
    """Export extended speech embedding with special tokens for CosyVoice3"""
    print("\n" + "=" * 60)
    print("Exporting CosyVoice3 Speech Embedding (with special tokens)")
    print("=" * 60)

    # Load config
    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        qwen_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': qwen_path})

    # Load LLM
    llm_config = configs['llm']
    llm_path = os.path.join(model_dir, 'llm.pt')
    state_dict = torch.load(llm_path, map_location='cpu')
    llm_config.load_state_dict(state_dict)
    llm_config.eval()

    # Speech embedding includes special tokens (SOS, EOS, TASK_ID, FILL)
    speech_embedding = llm_config.speech_embedding

    # Check dimensions
    num_tokens, embed_dim = speech_embedding.weight.shape
    print(f"  Token count: {num_tokens}")
    print(f"  Embedding dim: {embed_dim}")

    # Special token indices for CosyVoice3
    speech_token_size = 6561
    print(f"  Speech token size: {speech_token_size}")
    print(f"  SOS token ID: {speech_token_size}")
    print(f"  EOS token ID: {speech_token_size + 1}")
    print(f"  TASK_ID token: {speech_token_size + 2}")

    # Note: The existing llm_speech_embedding ONNX should work for CosyVoice3
    print("  Note: Use existing llm_speech_embedding_fp16.onnx for CosyVoice3 special tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'onnx')

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CosyVoice3 Additional ONNX Export for Unity")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output_dir}")

    # Export components
    export_text_embedding(args.model_dir, args.output_dir, args.fp16)
    export_speech_embedding_extended(args.model_dir, args.output_dir, args.fp16)
    export_vocabulary(args.model_dir, args.output_dir)
    export_source_generator(args.model_dir, args.output_dir, args.fp16)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print("\nFor Unity Sentis, you'll need:")
    print("1. text_embedding_fp32.onnx - Convert text token IDs to embeddings")
    print("2. qwen2_vocab.json - For C# BPE tokenizer implementation")
    print("3. Existing ONNX models (LLM, Flow, HiFT)")
    print("4. Custom STFT/ISTFT implementation in C# for HiFT source generation")


if __name__ == '__main__':
    main()
