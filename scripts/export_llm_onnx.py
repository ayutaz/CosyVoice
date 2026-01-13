#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
LLM (Qwen2) ONNX Export Script

This script exports the LLM components to ONNX format for Unity Sentis.

Export Strategy:
- speech_embedding: Embedding layer for speech tokens
- llm_decoder: Linear layer for logits output
- llm_backbone: Qwen2ForCausalLM (split into initial and with-cache versions)

For autoregressive inference in Unity:
1. Initial pass: Use llm_initial.onnx (no KV cache)
2. Subsequent passes: Use llm_decode.onnx (with KV cache)

Usage:
    python scripts/export_llm_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
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

# Disable SDPA for ONNX export compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  # Use math implementation instead

from hyperpyyaml import load_hyperpyyaml


class SpeechEmbeddingWrapper(nn.Module):
    """Wrapper for speech token embedding export"""

    def __init__(self, llm):
        super().__init__()
        self.speech_embedding = llm.speech_embedding

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token: [batch, 1] speech token index
        Returns:
            embedded: [batch, 1, embed_dim]
        """
        return self.speech_embedding(token)


class LLMDecoderWrapper(nn.Module):
    """Wrapper for LLM decoder (logits output) export"""

    def __init__(self, llm):
        super().__init__()
        self.llm_decoder = llm.llm_decoder

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, 1, hidden_dim] last hidden state
        Returns:
            logits: [batch, vocab_size] output logits
        """
        return self.llm_decoder(hidden_state[:, -1, :])


class LLMBackboneInitialWrapper(nn.Module):
    """
    Wrapper for initial LLM forward pass (with KV cache output).
    Used for the first token generation.

    Returns both hidden_states and KV cache for subsequent decode steps.
    """

    def __init__(self, qwen2_model, num_layers: int, num_heads: int, head_dim: int):
        super().__init__()
        self.model = qwen2_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple:
        """
        Args:
            inputs_embeds: [batch, seq_len, hidden_dim] input embeddings
            attention_mask: [batch, seq_len] attention mask (1 for valid, 0 for pad)
        Returns:
            hidden_states: [batch, seq_len, hidden_dim] output hidden states
            past_key_values_flat: [num_layers * 2, batch, num_heads, seq_len, head_dim] KV cache
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Flatten KV cache for ONNX
        past_key_values = outputs.past_key_values
        past_flat = []
        for i in range(self.num_layers):
            key, value = past_key_values[i]
            past_flat.append(key)
            past_flat.append(value)
        past_key_values_flat = torch.stack(past_flat, dim=0)

        return hidden_states, past_key_values_flat


class LLMBackboneDecodeWrapper(nn.Module):
    """
    Wrapper for LLM decode step (with KV cache).
    Used for subsequent token generation.

    Note: KV cache is flattened for ONNX compatibility.
    Uses DynamicCache from transformers for proper cache handling.
    """

    def __init__(self, qwen2_model, num_layers: int, num_heads: int, head_dim: int):
        super().__init__()
        self.model = qwen2_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values_flat: torch.Tensor
    ) -> tuple:
        """
        Args:
            inputs_embeds: [batch, 1, hidden_dim] current token embedding
            attention_mask: [batch, total_seq_len] attention mask
            past_key_values_flat: [num_layers * 2, batch, num_heads, past_len, head_dim]
        Returns:
            hidden_states: [batch, 1, hidden_dim] output hidden state
            new_past_key_values_flat: [num_layers * 2, batch, num_heads, new_len, head_dim]
        """
        from transformers.cache_utils import DynamicCache

        batch_size = inputs_embeds.shape[0]
        past_len = past_key_values_flat.shape[3]

        # Create DynamicCache from flattened tensor
        cache = DynamicCache()
        for i in range(self.num_layers):
            key = past_key_values_flat[i * 2]
            value = past_key_values_flat[i * 2 + 1]
            cache.update(key, value, i)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )

        hidden_states = outputs.hidden_states[-1]

        # Flatten new KV cache from DynamicCache
        new_cache = outputs.past_key_values
        new_past_flat = []
        for i in range(self.num_layers):
            key, value = new_cache[i]
            new_past_flat.append(key)
            new_past_flat.append(value)
        new_past_key_values_flat = torch.stack(new_past_flat, dim=0)

        return hidden_states, new_past_key_values_flat


def load_llm_model(model_dir: str, device: str = 'cpu'):
    """Load LLM model from model directory using AutoModel"""

    print(f"Loading model from {model_dir}")

    # Use the official AutoModel to load
    from cosyvoice.cli.cosyvoice import AutoModel

    # AutoModel handles loading properly
    cosyvoice = AutoModel(model_dir=model_dir)

    # Get the LLM from the loaded model
    llm = cosyvoice.model.llm

    llm.eval()
    llm.to(device)

    return llm, cosyvoice


def reload_qwen2_with_eager_attention(llm, model_dir: str):
    """Reload Qwen2 model with eager attention for ONNX compatibility"""

    from transformers import Qwen2Model, Qwen2Config
    import os
    import copy

    # Structure:
    # llm.llm = Qwen2Encoder
    # llm.llm.model = Qwen2ForCausalLM
    # llm.llm.model.model = Qwen2Model (the actual transformer)
    encoder = llm.llm  # This is Qwen2Encoder
    qwen2_for_causal = encoder.model  # This is Qwen2ForCausalLM
    qwen2_model = qwen2_for_causal.model  # This is Qwen2Model

    # Get the original model's state dict and config directly from qwen2_model
    original_state_dict = qwen2_model.state_dict()
    original_config = qwen2_model.config

    print(f"Original Qwen2 config: {original_config.hidden_size} hidden, {original_config.num_hidden_layers} layers")
    print(f"Original attn_implementation: {getattr(original_config, '_attn_implementation', 'unknown')}")
    print(f"Original state_dict keys: {len(original_state_dict)} keys")
    print(f"Sample keys: {list(original_state_dict.keys())[:5]}")

    # Create new config with eager attention
    new_config = copy.deepcopy(original_config)
    new_config._attn_implementation = "eager"

    print("Creating new Qwen2Model with eager attention...")

    # Create new model with eager attention
    new_model = Qwen2Model(new_config)

    # Load the original weights directly (they should match now)
    new_model.load_state_dict(original_state_dict)

    print("Successfully created Qwen2Model with eager attention")
    new_model.eval()

    return new_model


def export_speech_embedding(
    llm: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export speech embedding to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting Speech Embedding to ONNX")
    print(f"{'='*60}")

    wrapper = SpeechEmbeddingWrapper(llm)
    wrapper.eval()

    # Dummy input - single token
    batch_size = 1
    token = torch.zeros(batch_size, 1, dtype=torch.long)

    if fp16:
        wrapper = wrapper.half()

    print(f"Input shape: token {token.shape}")
    print(f"Vocab size: {llm.speech_embedding.num_embeddings}")
    print(f"Embedding dim: {llm.speech_embedding.embedding_dim}")

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


def export_llm_decoder(
    llm: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False
):
    """Export LLM decoder (logits) to ONNX"""

    print(f"\n{'='*60}")
    print("Exporting LLM Decoder to ONNX")
    print(f"{'='*60}")

    wrapper = LLMDecoderWrapper(llm)
    wrapper.eval()

    # Dummy input
    batch_size = 1
    hidden_dim = llm.llm_output_size
    hidden_state = torch.randn(batch_size, 1, hidden_dim)

    if fp16:
        wrapper = wrapper.half()
        hidden_state = hidden_state.half()

    print(f"Input shape: hidden_state {hidden_state.shape}")
    print(f"Output vocab size: {llm.llm_decoder.out_features}")

    with torch.no_grad():
        output = wrapper(hidden_state)
    print(f"Output shape: logits {output.shape}")

    torch.onnx.export(
        wrapper,
        (hidden_state,),
        output_path,
        opset_version=opset_version,
        input_names=['hidden_state'],
        output_names=['logits'],
        dynamic_axes={
            'hidden_state': {0: 'batch'},
            'logits': {0: 'batch'}
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


def export_llm_backbone_initial(
    llm: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False,
    eager_qwen2: nn.Module = None
):
    """Export LLM backbone initial pass (with KV cache output)"""

    print(f"\n{'='*60}")
    print("Exporting LLM Backbone (Initial Pass with KV Cache)")
    print(f"{'='*60}")

    # Get model config
    config = get_qwen2_config(llm)
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    head_dim = config['head_dim']

    print(f"Model config: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")

    # Use eager_qwen2 if provided (it's already Qwen2Model), otherwise use original
    if eager_qwen2 is not None:
        qwen2_model = eager_qwen2  # This is already a Qwen2Model
        print("Using eager attention Qwen2Model")
    else:
        qwen2_model = llm.llm.model
        print("Using original Qwen2Model (may use SDPA)")

    wrapper = LLMBackboneInitialWrapper(qwen2_model, num_layers, num_heads, head_dim)
    wrapper.eval()

    # Dummy input
    batch_size = 1
    seq_len = 32  # Typical initial sequence length
    hidden_dim = llm.llm_input_size

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    if fp16:
        wrapper = wrapper.half()
        inputs_embeds = inputs_embeds.half()

    print(f"Input shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    with torch.no_grad():
        hidden_states, past_kv = wrapper(inputs_embeds, attention_mask)
    print(f"Output shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  past_key_values: {past_kv.shape}")

    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask),
        output_path,
        opset_version=opset_version,
        input_names=['inputs_embeds', 'attention_mask'],
        output_names=['hidden_states', 'past_key_values'],
        dynamic_axes={
            'inputs_embeds': {0: 'batch', 1: 'seq_len'},
            'attention_mask': {0: 'batch', 1: 'seq_len'},
            'hidden_states': {0: 'batch', 1: 'seq_len'},
            'past_key_values': {1: 'batch', 3: 'seq_len'}
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


def get_qwen2_config(llm):
    """Get Qwen2 model configuration"""
    config = llm.llm.model.config
    return {
        'num_layers': config.num_hidden_layers,
        'num_heads': config.num_key_value_heads,  # For GQA
        'head_dim': config.hidden_size // config.num_attention_heads,
        'hidden_size': config.hidden_size,
    }


def export_llm_backbone_decode(
    llm: nn.Module,
    output_path: str,
    opset_version: int = 15,
    fp16: bool = False,
    eager_qwen2: nn.Module = None
):
    """Export LLM backbone decode step (with KV cache)"""

    print(f"\n{'='*60}")
    print("Exporting LLM Backbone (Decode Step with KV Cache)")
    print(f"{'='*60}")

    config = get_qwen2_config(llm)
    print(f"Qwen2 config: {config}")

    # Use eager_qwen2 if provided (it's already Qwen2Model), otherwise use original
    if eager_qwen2 is not None:
        qwen2_model = eager_qwen2  # This is already a Qwen2Model
        print("Using eager attention Qwen2Model")
    else:
        qwen2_model = llm.llm.model
        print("Using original Qwen2Model (may use SDPA)")
    wrapper = LLMBackboneDecodeWrapper(
        qwen2_model,
        config['num_layers'],
        config['num_heads'],
        config['head_dim']
    )
    wrapper.eval()

    # Dummy input
    batch_size = 1
    past_len = 32
    hidden_dim = llm.llm_input_size

    inputs_embeds = torch.randn(batch_size, 1, hidden_dim)
    attention_mask = torch.ones(batch_size, past_len + 1)
    past_key_values_flat = torch.randn(
        config['num_layers'] * 2,
        batch_size,
        config['num_heads'],
        past_len,
        config['head_dim']
    )

    if fp16:
        wrapper = wrapper.half()
        inputs_embeds = inputs_embeds.half()
        past_key_values_flat = past_key_values_flat.half()

    print(f"Input shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  past_key_values_flat: {past_key_values_flat.shape}")

    try:
        with torch.no_grad():
            hidden_states, new_cache = wrapper(inputs_embeds, attention_mask, past_key_values_flat)
        print(f"Output shapes:")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  new_cache: {new_cache.shape}")

        torch.onnx.export(
            wrapper,
            (inputs_embeds, attention_mask, past_key_values_flat),
            output_path,
            opset_version=opset_version,
            input_names=['inputs_embeds', 'attention_mask', 'past_key_values'],
            output_names=['hidden_states', 'new_past_key_values'],
            dynamic_axes={
                'inputs_embeds': {0: 'batch'},
                'attention_mask': {0: 'batch', 1: 'total_len'},
                'past_key_values': {1: 'batch', 3: 'past_len'},
                'hidden_states': {0: 'batch'},
                'new_past_key_values': {1: 'batch', 3: 'new_len'}
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

    except Exception as e:
        print(f"Error exporting decode step: {e}")
        print("This is expected for complex models. Unity will need custom KV cache handling.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export LLM components to ONNX')
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
        '--skip_backbone',
        action='store_true',
        help='Skip backbone export (large model)'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'onnx')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Opset version: {args.opset_version}")
    print(f"FP16: {args.fp16}")

    # Load model
    llm, cosyvoice = load_llm_model(args.model_dir)

    # Print model info
    print(f"\nLLM Configuration:")
    print(f"  - Input size: {llm.llm_input_size}")
    print(f"  - Output size: {llm.llm_output_size}")
    print(f"  - Speech token size: {llm.speech_token_size}")

    config = get_qwen2_config(llm)
    print(f"  - Qwen2 layers: {config['num_layers']}")
    print(f"  - Qwen2 heads (KV): {config['num_heads']}")
    print(f"  - Qwen2 head dim: {config['head_dim']}")
    print(f"  - Qwen2 hidden size: {config['hidden_size']}")

    # Export components
    precision_suffix = '_fp16' if args.fp16 else '_fp32'

    # 1. Speech Embedding
    embed_path = os.path.join(args.output_dir, f'llm_speech_embedding{precision_suffix}.onnx')
    export_speech_embedding(llm, embed_path, args.opset_version, args.fp16)

    # 2. LLM Decoder
    decoder_path = os.path.join(args.output_dir, f'llm_decoder{precision_suffix}.onnx')
    export_llm_decoder(llm, decoder_path, args.opset_version, args.fp16)

    if not args.skip_backbone:
        # Reload Qwen2 with eager attention for backbone export
        print("\n" + "="*60)
        print("Reloading Qwen2 with eager attention for ONNX export")
        print("="*60)
        eager_qwen2 = reload_qwen2_with_eager_attention(llm, args.model_dir)
        eager_qwen2.eval()

        # 3. LLM Backbone Initial
        initial_path = os.path.join(args.output_dir, f'llm_backbone_initial{precision_suffix}.onnx')
        export_llm_backbone_initial(llm, initial_path, args.opset_version, args.fp16, eager_qwen2)

        # 4. LLM Backbone Decode (with KV cache)
        decode_path = os.path.join(args.output_dir, f'llm_backbone_decode{precision_suffix}.onnx')
        export_llm_backbone_decode(llm, decode_path, args.opset_version, args.fp16, eager_qwen2)

    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"Speech Embedding: {embed_path}")
    print(f"LLM Decoder: {decoder_path}")
    if not args.skip_backbone:
        print(f"LLM Backbone Initial: {initial_path}")
        print(f"LLM Backbone Decode: {decode_path}")

    print(f"\n{'='*60}")
    print("Unity Implementation Notes")
    print(f"{'='*60}")
    print("""
For Unity Sentis LLM inference:

1. Prepare input embeddings:
   - Concatenate: [SOS_emb, text_emb, task_id_emb]
   - Use speech_embedding for previous generated tokens

2. Initial forward pass:
   - Use llm_backbone_initial.onnx
   - No KV cache needed

3. Autoregressive generation:
   - Use llm_backbone_decode.onnx with KV cache
   - For each step:
     a. Pass current token embedding
     b. Update KV cache
     c. Get logits from llm_decoder
     d. Sample next token

4. Sampling:
   - Apply Top-K sampling (K=25 recommended)
   - Stop on EOS token

KV Cache Shape: [num_layers * 2, batch, num_kv_heads, seq_len, head_dim]
""")


if __name__ == '__main__':
    main()
