#!/usr/bin/env python3
"""Upload CosyVoice3 ONNX models to Hugging Face Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Repository configuration
REPO_ID = "ayousanz/cosy-voice3-onnx"
MODEL_DIR = Path("pretrained_models/Fun-CosyVoice3-0.5B")
ONNX_DIR = MODEL_DIR / "onnx"

# Files to upload
ONNX_FILES = [
    # From onnx subdirectory
    (ONNX_DIR / "README.md", "README.md"),
    (ONNX_DIR / "text_embedding_fp32.onnx", "text_embedding_fp32.onnx"),
    (ONNX_DIR / "llm_backbone_initial_fp16.onnx", "llm_backbone_initial_fp16.onnx"),
    (ONNX_DIR / "llm_backbone_decode_fp16.onnx", "llm_backbone_decode_fp16.onnx"),
    (ONNX_DIR / "llm_decoder_fp16.onnx", "llm_decoder_fp16.onnx"),
    (ONNX_DIR / "llm_speech_embedding_fp16.onnx", "llm_speech_embedding_fp16.onnx"),
    (ONNX_DIR / "flow_token_embedding_fp16.onnx", "flow_token_embedding_fp16.onnx"),
    (ONNX_DIR / "flow_pre_lookahead_fp16.onnx", "flow_pre_lookahead_fp16.onnx"),
    (ONNX_DIR / "flow_speaker_projection_fp16.onnx", "flow_speaker_projection_fp16.onnx"),
    (ONNX_DIR / "flow.decoder.estimator.fp16.onnx", "flow.decoder.estimator.fp16.onnx"),
    (ONNX_DIR / "hift_f0_predictor_fp32.onnx", "hift_f0_predictor_fp32.onnx"),
    (ONNX_DIR / "hift_source_generator_fp32.onnx", "hift_source_generator_fp32.onnx"),
    (ONNX_DIR / "hift_decoder_fp32.onnx", "hift_decoder_fp32.onnx"),
    # From parent directory (original model files)
    (MODEL_DIR / "campplus.onnx", "campplus.onnx"),
    (MODEL_DIR / "speech_tokenizer_v3.onnx", "speech_tokenizer_v3.onnx"),
]


def main():
    # Check all files exist
    print("Checking files...")
    missing = []
    for local_path, _ in ONNX_FILES:
        if not local_path.exists():
            missing.append(str(local_path))
        else:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {local_path.name} ({size_mb:.1f} MB)")

    if missing:
        print(f"\nMissing files: {missing}")
        return

    print(f"\nUploading to {REPO_ID}...")

    # Initialize API
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository {REPO_ID} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Upload files
    for local_path, remote_path in ONNX_FILES:
        print(f"Uploading {local_path.name}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  [OK] Uploaded {remote_path}")
        except Exception as e:
            print(f"  [FAIL] {e}")

    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
