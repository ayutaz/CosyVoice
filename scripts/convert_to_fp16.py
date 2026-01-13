#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Convert ONNX Models to FP16

This script converts all exported ONNX models from FP32 to FP16 precision.
FP16 reduces model size by ~50% with minimal quality loss for inference.

Usage:
    python scripts/convert_to_fp16.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

import argparse
import os
import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16


def convert_to_fp16_ort(
    input_path: str,
    output_path: str,
    keep_io_types: bool = True,
    op_block_list: list = None
) -> bool:
    """Convert ONNX model from FP32 to FP16 using onnxruntime.transformers

    This is needed for large models like LLM backbone that fail with onnxconverter_common.
    """
    from onnxruntime.transformers import float16 as ort_float16

    if op_block_list is None:
        op_block_list = [
            'LayerNormalization',
            'SimplifiedLayerNormalization',
            'ReduceMean',
            'Softmax',
            'ReduceSum',
        ]

    try:
        print(f"  Loading: {input_path}")
        model = onnx.load(input_path)

        print(f"  Converting to FP16 (onnxruntime)...")
        model_fp16 = ort_float16.convert_float_to_float16(
            model,
            keep_io_types=keep_io_types,
            op_block_list=op_block_list,
        )

        print(f"  Saving: {output_path}")
        onnx.save(model_fp16, output_path)

        # Check sizes
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - output_size / input_size) * 100

        print(f"  Size: {input_size:.1f} MB -> {output_size:.1f} MB ({reduction:.1f}% reduction)")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def convert_to_fp16(
    input_path: str,
    output_path: str,
    keep_io_types: bool = True,
    op_block_list: list = None
) -> bool:
    """Convert ONNX model from FP32 to FP16"""

    if op_block_list is None:
        # Operations that should stay in FP32 for numerical stability
        op_block_list = [
            'LayerNormalization',
            'ReduceMean',
            'Softmax',
            'ReduceSum',
        ]

    try:
        print(f"  Loading: {input_path}")
        model = onnx.load(input_path)

        print(f"  Converting to FP16...")
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=keep_io_types,
            op_block_list=op_block_list,
        )

        print(f"  Saving: {output_path}")
        onnx.save(model_fp16, output_path)

        # Check sizes
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - output_size / input_size) * 100

        print(f"  Size: {input_size:.1f} MB -> {output_size:.1f} MB ({reduction:.1f}% reduction)")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def convert_all_models(model_dir: str, output_dir: str = None):
    """Convert all ONNX models in the model directory to FP16"""

    onnx_dir = os.path.join(model_dir, 'onnx')

    if output_dir is None:
        output_dir = onnx_dir

    os.makedirs(output_dir, exist_ok=True)

    # List of models to convert (using standard onnxconverter_common)
    models_to_convert = [
        # Exported models (in onnx/)
        ('flow_token_embedding_fp32.onnx', 'flow_token_embedding_fp16.onnx', onnx_dir),
        ('flow_speaker_projection_fp32.onnx', 'flow_speaker_projection_fp16.onnx', onnx_dir),
        ('flow_pre_lookahead_fp32.onnx', 'flow_pre_lookahead_fp16.onnx', onnx_dir),
        ('llm_speech_embedding_fp32.onnx', 'llm_speech_embedding_fp16.onnx', onnx_dir),
        ('llm_decoder_fp32.onnx', 'llm_decoder_fp16.onnx', onnx_dir),
    ]

    # Models that need onnxruntime.transformers converter
    # (onnxconverter_common fails with type errors or "'list' object has no attribute 'input'" error)
    ort_models = [
        ('hift_f0_predictor_fp32.onnx', 'hift_f0_predictor_fp16.onnx', onnx_dir),
        ('hift_decoder_fp32.onnx', 'hift_decoder_fp16.onnx', onnx_dir),
        ('llm_backbone_initial_fp32.onnx', 'llm_backbone_initial_fp16.onnx', onnx_dir),
        ('llm_backbone_decode_fp32.onnx', 'llm_backbone_decode_fp16.onnx', onnx_dir),
    ]

    # Pre-existing models (in model_dir root)
    pre_existing = [
        ('campplus.onnx', 'campplus_fp16.onnx', model_dir),
        ('flow.decoder.estimator.fp32.onnx', 'flow.decoder.estimator.fp16.onnx', model_dir),
        ('speech_tokenizer_v3.onnx', 'speech_tokenizer_v3_fp16.onnx', model_dir),
    ]

    results = {}

    print(f"\n{'='*60}")
    print("Converting Exported Models to FP16")
    print(f"{'='*60}\n")

    for input_name, output_name, source_dir in models_to_convert:
        input_path = os.path.join(source_dir, input_name)
        output_path = os.path.join(output_dir, output_name)

        if not os.path.exists(input_path):
            print(f"[SKIP] {input_name}: Not found")
            results[input_name] = 'SKIP'
            continue

        print(f"[CONVERTING] {input_name}")
        success = convert_to_fp16(input_path, output_path)
        results[input_name] = 'OK' if success else 'FAIL'
        print()

    print(f"\n{'='*60}")
    print("Converting Models to FP16 (using onnxruntime.transformers)")
    print(f"{'='*60}\n")

    for input_name, output_name, source_dir in ort_models:
        input_path = os.path.join(source_dir, input_name)
        output_path = os.path.join(output_dir, output_name)

        if not os.path.exists(input_path):
            print(f"[SKIP] {input_name}: Not found")
            results[input_name] = 'SKIP'
            continue

        print(f"[CONVERTING] {input_name}")
        success = convert_to_fp16_ort(input_path, output_path)
        results[input_name] = 'OK' if success else 'FAIL'
        print()

    print(f"\n{'='*60}")
    print("Converting Pre-existing Models to FP16")
    print(f"{'='*60}\n")

    for input_name, output_name, source_dir in pre_existing:
        input_path = os.path.join(source_dir, input_name)
        output_path = os.path.join(output_dir, output_name)

        if not os.path.exists(input_path):
            print(f"[SKIP] {input_name}: Not found")
            results[input_name] = 'SKIP'
            continue

        print(f"[CONVERTING] {input_name}")
        success = convert_to_fp16(input_path, output_path)
        results[input_name] = 'OK' if success else 'FAIL'
        print()

    return results


def verify_fp16_models(model_dir: str):
    """Verify converted FP16 models work correctly"""

    import onnxruntime as ort
    import numpy as np

    onnx_dir = os.path.join(model_dir, 'onnx')

    print(f"\n{'='*60}")
    print("Verifying FP16 Models")
    print(f"{'='*60}\n")

    fp16_models = [
        ('hift_f0_predictor_fp16.onnx', {'mel': (1, 80, 100)}),
        ('flow_token_embedding_fp16.onnx', {'token': (1, 100)}),
        ('flow_speaker_projection_fp16.onnx', {'embedding': (1, 192)}),
        ('flow_pre_lookahead_fp16.onnx', {'token_embedded': (1, 100, 80)}),
        ('llm_speech_embedding_fp16.onnx', {'token': (1, 1)}),
        ('llm_decoder_fp16.onnx', {'hidden_state': (1, 1, 896)}),
        ('llm_backbone_initial_fp16.onnx', {'inputs_embeds': (1, 32, 896), 'attention_mask': (1, 32)}),
        ('llm_backbone_decode_fp16.onnx', {
            'inputs_embeds': (1, 1, 896),
            'attention_mask': (1, 33),
            'past_key_values': (48, 1, 2, 32, 64)
        }),
    ]

    for filename, inputs_spec in fp16_models:
        path = os.path.join(onnx_dir, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename}: Not found")
            continue

        try:
            session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

            # Create dummy inputs based on actual model input types
            inputs = {}
            for input_info in session.get_inputs():
                name = input_info.name
                if name in inputs_spec:
                    shape = inputs_spec[name]
                    if 'int' in input_info.type.lower():
                        inputs[name] = np.zeros(shape, dtype=np.int64)
                    else:
                        inputs[name] = np.random.randn(*shape).astype(np.float32)

            # Run inference
            outputs = session.run(None, inputs)
            print(f"[OK] {filename}: {[o.shape for o in outputs]}")

        except Exception as e:
            print(f"[FAIL] {filename}: {e}")


def print_summary(model_dir: str):
    """Print summary of FP16 conversion"""

    onnx_dir = os.path.join(model_dir, 'onnx')

    print(f"\n{'='*60}")
    print("FP16 Conversion Summary")
    print(f"{'='*60}\n")

    total_fp32 = 0
    total_fp16 = 0

    files = [
        ('hift_f0_predictor', onnx_dir),
        ('hift_decoder', onnx_dir),
        ('flow_token_embedding', onnx_dir),
        ('flow_speaker_projection', onnx_dir),
        ('flow_pre_lookahead', onnx_dir),
        ('llm_speech_embedding', onnx_dir),
        ('llm_decoder', onnx_dir),
        ('llm_backbone_initial', onnx_dir),
        ('llm_backbone_decode', onnx_dir),
    ]

    pre_existing = [
        ('campplus', model_dir, onnx_dir),
        ('flow.decoder.estimator', model_dir, onnx_dir),
        ('speech_tokenizer_v3', model_dir, onnx_dir),
    ]

    print("Exported Models:")
    print("-" * 60)
    for name, source_dir in files:
        fp32_path = os.path.join(source_dir, f'{name}_fp32.onnx')
        fp16_path = os.path.join(source_dir, f'{name}_fp16.onnx')

        if os.path.exists(fp32_path) and os.path.exists(fp16_path):
            fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
            fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
            total_fp32 += fp32_size
            total_fp16 += fp16_size
            print(f"  {name}: {fp32_size:.1f} MB -> {fp16_size:.1f} MB")
        elif os.path.exists(fp32_path):
            fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
            total_fp32 += fp32_size
            print(f"  {name}: {fp32_size:.1f} MB (FP16 not converted)")

    print("\nPre-existing Models:")
    print("-" * 60)
    for name, source_dir, output_dir in pre_existing:
        if name == 'flow.decoder.estimator':
            fp32_path = os.path.join(source_dir, f'{name}.fp32.onnx')
            fp16_path = os.path.join(output_dir, f'{name}.fp16.onnx')
        else:
            fp32_path = os.path.join(source_dir, f'{name}.onnx')
            fp16_path = os.path.join(output_dir, f'{name}_fp16.onnx')

        if os.path.exists(fp32_path):
            fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
            total_fp32 += fp32_size
            if os.path.exists(fp16_path):
                fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
                total_fp16 += fp16_size
                print(f"  {name}: {fp32_size:.1f} MB -> {fp16_size:.1f} MB")
            else:
                print(f"  {name}: {fp32_size:.1f} MB (FP16 not converted)")

    print("\n" + "-" * 60)
    print(f"Total FP32: {total_fp32:.1f} MB")
    print(f"Total FP16: {total_fp16:.1f} MB")
    if total_fp32 > 0:
        reduction = (1 - total_fp16 / total_fp32) * 100
        print(f"Reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to FP16')
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
        help='Output directory for FP16 models (default: same as input)'
    )
    parser.add_argument(
        '--skip_large',
        action='store_true',
        help='Skip large models (flow estimator, speech tokenizer)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify FP16 models after conversion'
    )

    args = parser.parse_args()

    print(f"Model directory: {args.model_dir}")

    # Convert models
    results = convert_all_models(args.model_dir, args.output_dir)

    # Verify if requested
    if args.verify:
        verify_fp16_models(args.model_dir)

    # Print summary
    print_summary(args.model_dir)

    # Final status
    print(f"\n{'='*60}")
    print("Conversion Results")
    print(f"{'='*60}")
    ok_count = sum(1 for v in results.values() if v == 'OK')
    fail_count = sum(1 for v in results.values() if v == 'FAIL')
    skip_count = sum(1 for v in results.values() if v == 'SKIP')
    print(f"  OK: {ok_count}, FAIL: {fail_count}, SKIP: {skip_count}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
