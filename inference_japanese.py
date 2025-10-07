#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple command-line TTS inference script for Japanese
Supports Japanese hybrid preprocessing with pyopenjtalk-plus

Usage:
    python inference_japanese.py --text "今日は良い天気です。" --model_dir pretrained_models/CosyVoice2-0.5B
    python inference_japanese.py --text "こんにちは" --model_dir pretrained_models/CosyVoice2-0.5B --output output.wav
"""

import argparse
import sys
import os

# Add third_party path
sys.path.append('third_party/Matcha-TTS')

def main():
    parser = argparse.ArgumentParser(description='CosyVoice Japanese TTS Inference')
    parser.add_argument('--text', type=str, required=True,
                        help='Text to synthesize (Japanese or other languages)')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to pretrained model directory')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output audio file path (default: output.wav)')
    parser.add_argument('--spk_id', type=str, default='中文女',
                        help='Speaker ID for SFT mode (default: 中文女)')
    parser.add_argument('--use_japanese', action='store_true', default=True,
                        help='Enable Japanese frontend preprocessing (default: True)')
    parser.add_argument('--use_hybrid', action='store_true', default=True,
                        help='Enable hybrid mode with kabosu-core (default: True)')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming mode')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed (default: 1.0)')

    args = parser.parse_args()

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' does not exist!")
        print()
        print("Please download models first:")
        print("  python download_models.py")
        print()
        print("Or manually download using:")
        print("  from modelscope import snapshot_download")
        print("  snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')")
        sys.exit(1)

    # Import required modules
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        import torchaudio
        import torch
        print("✓ CosyVoice modules loaded")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Check Japanese frontend availability
    if args.use_japanese:
        try:
            import pyopenjtalk
            print("✓ pyopenjtalk-plus is available")
        except ImportError:
            print("⚠ pyopenjtalk-plus is NOT available, disabling Japanese frontend")
            args.use_japanese = False

        if args.use_hybrid:
            try:
                from kabosu_core import Kabosu
                print("✓ kabosu-core is available (hybrid mode)")
            except ImportError:
                print("⚠ kabosu-core is NOT available, using pyopenjtalk-only mode")
                args.use_hybrid = False

    print()
    print("="*60)
    print("CosyVoice Japanese TTS Inference")
    print("="*60)
    print(f"Model: {args.model_dir}")
    print(f"Text: {args.text}")
    print(f"Output: {args.output}")
    print(f"Japanese frontend: {args.use_japanese}")
    print(f"Hybrid mode: {args.use_hybrid}")
    print("="*60)
    print()

    # Determine model type
    if 'CosyVoice2' in args.model_dir:
        print("Loading CosyVoice2 model...")
        cosyvoice = CosyVoice2(
            args.model_dir,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        model_type = 'cosyvoice2'
    else:
        print("Loading CosyVoice model...")
        cosyvoice = CosyVoice(
            args.model_dir,
            load_jit=False,
            load_trt=False,
            fp16=False
        )
        model_type = 'cosyvoice'

    print(f"✓ Model loaded (sample_rate: {cosyvoice.sample_rate} Hz)")
    print()

    # Update frontend with Japanese support
    if args.use_japanese:
        print("Enabling Japanese frontend preprocessing...")
        cosyvoice.frontend.use_japanese_frontend = True
        cosyvoice.frontend.use_hybrid = args.use_hybrid
        if args.use_hybrid:
            try:
                from kabosu_core import Kabosu
                cosyvoice.frontend.kabosu = Kabosu()
                print("✓ Hybrid mode enabled (kabosu-core + pyopenjtalk-plus)")
            except Exception as e:
                print(f"⚠ Failed to initialize kabosu-core: {e}")
                print("  Falling back to pyopenjtalk-only mode")
                cosyvoice.frontend.use_hybrid = False
        else:
            print("✓ pyopenjtalk-only mode enabled")
        print()

    # Synthesize speech
    print("Synthesizing speech...")
    try:
        # Use zero-shot mode with default speaker
        # For CosyVoice2, we can use inference_sft if speaker exists
        output_chunks = []

        if model_type == 'cosyvoice2':
            # CosyVoice2 typically uses zero-shot or instruct mode
            # For simple TTS, we'll use a basic approach
            print("Using CosyVoice2 synthesis...")

            # Get list of available speakers
            spk_info = cosyvoice.list_available_spks()
            if args.spk_id in spk_info:
                print(f"Using speaker: {args.spk_id}")
                for i, chunk in enumerate(cosyvoice.inference_sft(
                    args.text,
                    args.spk_id,
                    stream=args.stream,
                    speed=args.speed
                )):
                    output_chunks.append(chunk['tts_speech'])
            else:
                print(f"Speaker '{args.spk_id}' not found.")
                print(f"Available speakers: {', '.join(spk_info[:10])}")
                if spk_info:
                    first_spk = spk_info[0]
                    print(f"Using first available speaker: {first_spk}")
                    for i, chunk in enumerate(cosyvoice.inference_sft(
                        args.text,
                        first_spk,
                        stream=args.stream,
                        speed=args.speed
                    )):
                        output_chunks.append(chunk['tts_speech'])
                else:
                    print("No speakers available! Cannot synthesize.")
                    sys.exit(1)
        else:
            # CosyVoice (300M models)
            print("Using CosyVoice synthesis...")
            spk_info = cosyvoice.list_available_spks()
            if args.spk_id in spk_info:
                for i, chunk in enumerate(cosyvoice.inference_sft(
                    args.text,
                    args.spk_id,
                    stream=args.stream
                )):
                    output_chunks.append(chunk['tts_speech'])
            else:
                first_spk = spk_info[0] if spk_info else None
                if first_spk:
                    print(f"Using speaker: {first_spk}")
                    for i, chunk in enumerate(cosyvoice.inference_sft(
                        args.text,
                        first_spk,
                        stream=args.stream
                    )):
                        output_chunks.append(chunk['tts_speech'])
                else:
                    print("No speakers available!")
                    sys.exit(1)

        # Concatenate chunks
        if output_chunks:
            audio = torch.cat(output_chunks, dim=1)
            print(f"✓ Synthesis complete ({audio.shape[1] / cosyvoice.sample_rate:.2f} seconds)")

            # Save audio
            torchaudio.save(args.output, audio, cosyvoice.sample_rate)
            print(f"✓ Audio saved to: {args.output}")
        else:
            print("✗ No audio generated")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("="*60)
    print("Inference Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
