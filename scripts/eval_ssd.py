#!/usr/bin/env python3
"""Evaluation pipeline for Speech Speculative Decoding (SSD).

Reproduces Table 2 & 3 from:
  "Accelerating Autoregressive Speech Synthesis Inference With
   Speech Speculative Decoding" (Interspeech 2025)

Measures:
  - WER: via Whisper-large-v3 transcription
  - LM-RTF: LLM forward time / generated audio duration
  - Acceptance rate: accepted / total drafted tokens

Compares 3 models:
  1. CosyVoice2 (AR baseline)
  2. Draft model only (8-layer)
  3. CosyVoice2 + SSD

Usage:
    python scripts/eval_ssd.py \
        --model_dir pretrained_models/CosyVoice2-0.5B \
        --draft_model pretrained_models/CosyVoice2-0.5B/llm_draft.pt \
        --eval_data path/to/eval_list.txt \
        --beta_values 0.0,0.1,0.2,0.3,0.4 \
        --output_dir results/ssd_eval

    eval_list.txt format (tab-separated):
        <utterance_id>\t<text>\t<prompt_wav_path>
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SSD for CosyVoice2")
    parser.add_argument("--model_dir", required=True, help="CosyVoice2 model directory")
    parser.add_argument("--draft_model", default=None, help="Path to draft model (llm_draft.pt)")
    parser.add_argument("--eval_data", required=True, help="Evaluation list file")
    parser.add_argument("--beta_values", default="0.0,0.1,0.2,0.3,0.4",
                        help="Comma-separated tolerance (beta) values to evaluate")
    parser.add_argument("--num_draft_tokens", type=int, default=3, help="Number of draft tokens (Ld)")
    parser.add_argument("--output_dir", default="results/ssd_eval", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--whisper_model", default="large-v3", help="Whisper model for WER")
    parser.add_argument("--compute_wer", action="store_true", help="Compute WER using Whisper")
    return parser.parse_args()


def load_eval_data(eval_data_path, max_samples=-1):
    """Load evaluation list.

    Format: <utt_id>\t<text>\t<prompt_wav_path>
    """
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                samples.append({
                    "utt_id": parts[0],
                    "text": parts[1],
                    "prompt_wav": parts[2],
                })
            elif len(parts) == 2:
                samples.append({
                    "utt_id": parts[0],
                    "text": parts[1],
                    "prompt_wav": None,
                })
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def synthesize_and_measure(cosyvoice, samples, mode="ar", output_dir=None, sample_rate=24000):
    """Run inference and measure LM-RTF.

    Args:
        cosyvoice: CosyVoice2 instance.
        samples: list of eval samples.
        mode: "ar", "draft_only", or "ssd".
        output_dir: directory to save wav files.
        sample_rate: audio sample rate.

    Returns:
        list of dicts with utt_id, text, wav_path, lm_rtf, audio_duration.
    """
    results = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(samples):
        utt_id = sample["utt_id"]
        text = sample["text"]
        prompt_wav = sample.get("prompt_wav")

        lm_start = time.time()

        # Use cross_lingual mode (requires prompt_wav) or sft mode
        audio_chunks = []
        if prompt_wav is not None:
            for output in cosyvoice.inference_cross_lingual(text, prompt_wav, text_frontend=False):
                audio_chunks.append(output["tts_speech"])
        else:
            # Fallback: use first available speaker
            spks = cosyvoice.list_available_spks()
            if spks:
                for output in cosyvoice.inference_sft(text, spks[0], text_frontend=False):
                    audio_chunks.append(output["tts_speech"])
            else:
                print("Warning: no speakers and no prompt_wav for {}".format(utt_id))
                continue

        lm_elapsed = time.time() - lm_start

        if not audio_chunks:
            print("Warning: no audio generated for {}".format(utt_id))
            continue

        audio = torch.cat(audio_chunks, dim=-1)
        audio_duration = audio.shape[-1] / sample_rate
        lm_rtf = lm_elapsed / max(audio_duration, 1e-9)

        wav_path = None
        if output_dir:
            wav_path = os.path.join(output_dir, "{}.wav".format(utt_id))
            torchaudio.save(wav_path, audio.cpu(), sample_rate)

        results.append({
            "utt_id": utt_id,
            "text": text,
            "wav_path": wav_path,
            "lm_rtf": lm_rtf,
            "audio_duration": audio_duration,
            "lm_time": lm_elapsed,
        })

        if (idx + 1) % 10 == 0:
            print("  Processed {}/{} utterances".format(idx + 1, len(samples)))

    return results


def compute_wer_whisper(results, whisper_model_name="large-v3"):
    """Compute WER using Whisper transcription.

    Requires: pip install openai-whisper jiwer
    """
    try:
        import whisper
        from jiwer import wer as compute_wer_score
    except ImportError:
        print("Error: install openai-whisper and jiwer for WER computation")
        print("  pip install openai-whisper jiwer")
        return None

    print("Loading Whisper {} model...".format(whisper_model_name))
    model = whisper.load_model(whisper_model_name)

    hypotheses = []
    references = []
    for r in results:
        if r.get("wav_path") is None:
            continue
        result = model.transcribe(r["wav_path"], language="en")
        hyp = result["text"].strip().lower()
        ref = r["text"].strip().lower()
        hypotheses.append(hyp)
        references.append(ref)
        r["hypothesis"] = hyp

    if not hypotheses:
        return None

    wer_score = compute_wer_score(references, hypotheses)
    return wer_score


def run_evaluation(args):
    from cosyvoice.cli.cosyvoice import CosyVoice2

    samples = load_eval_data(args.eval_data, args.max_samples)
    print("Loaded {} evaluation samples".format(len(samples)))

    beta_values = [float(b) for b in args.beta_values.split(",")]
    all_results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. AR Baseline ---
    print("\n=== AR Baseline ===")
    cosyvoice_ar = CosyVoice2(args.model_dir)
    ar_dir = os.path.join(args.output_dir, "ar")
    ar_results = synthesize_and_measure(cosyvoice_ar, samples, mode="ar",
                                         output_dir=ar_dir, sample_rate=cosyvoice_ar.sample_rate)
    ar_rtfs = [r["lm_rtf"] for r in ar_results]
    print("AR: mean LM-RTF = {:.4f} ({} utterances)".format(np.mean(ar_rtfs), len(ar_rtfs)))

    ar_wer = None
    if args.compute_wer:
        ar_wer = compute_wer_whisper(ar_results, args.whisper_model)
        if ar_wer is not None:
            print("AR: WER = {:.2f}%".format(ar_wer * 100))

    all_results["ar"] = {
        "mean_lm_rtf": float(np.mean(ar_rtfs)),
        "wer": float(ar_wer) if ar_wer is not None else None,
        "num_utterances": len(ar_results),
        "results": ar_results,
    }
    del cosyvoice_ar
    torch.cuda.empty_cache()

    # --- 2. SSD with different beta values ---
    if args.draft_model and os.path.exists(args.draft_model):
        for beta in beta_values:
            print("\n=== SSD (beta={}) ===".format(beta))
            cosyvoice_ssd = CosyVoice2(args.model_dir, load_ssd=True,
                                        ssd_num_draft=args.num_draft_tokens,
                                        ssd_tolerance=beta)

            # Override tolerance if model is already loaded
            if hasattr(cosyvoice_ssd.model.llm, 'ssd') and cosyvoice_ssd.model.llm.ssd is not None:
                cosyvoice_ssd.model.llm.ssd.tolerance = beta
                cosyvoice_ssd.model.llm.ssd.reset_stats()

            ssd_dir = os.path.join(args.output_dir, "ssd_beta{}".format(beta))
            ssd_results = synthesize_and_measure(cosyvoice_ssd, samples, mode="ssd",
                                                  output_dir=ssd_dir, sample_rate=cosyvoice_ssd.sample_rate)
            ssd_rtfs = [r["lm_rtf"] for r in ssd_results]
            print("SSD (beta={}): mean LM-RTF = {:.4f}".format(beta, np.mean(ssd_rtfs)))

            # Get SSD stats
            ssd_stats = None
            if hasattr(cosyvoice_ssd.model.llm, 'ssd') and cosyvoice_ssd.model.llm.ssd is not None:
                ssd_stats = cosyvoice_ssd.model.llm.ssd.get_stats(
                    token_frame_rate=25, sample_rate=cosyvoice_ssd.sample_rate)
                print("SSD (beta={}): acceptance_rate = {:.2f}%, lm_rtf(stats) = {:.4f}".format(
                    beta, ssd_stats["acceptance_rate"] * 100, ssd_stats["lm_rtf"]))

            ssd_wer = None
            if args.compute_wer:
                ssd_wer = compute_wer_whisper(ssd_results, args.whisper_model)
                if ssd_wer is not None:
                    print("SSD (beta={}): WER = {:.2f}%".format(beta, ssd_wer * 100))

            all_results["ssd_beta{}".format(beta)] = {
                "beta": beta,
                "num_draft_tokens": args.num_draft_tokens,
                "mean_lm_rtf": float(np.mean(ssd_rtfs)),
                "wer": float(ssd_wer) if ssd_wer is not None else None,
                "ssd_stats": ssd_stats,
                "num_utterances": len(ssd_results),
                "results": ssd_results,
            }
            del cosyvoice_ssd
            torch.cuda.empty_cache()

    # --- 3. Summary ---
    print("\n=== Summary ===")
    print("{:<25} {:>10} {:>10} {:>15}".format("Model", "LM-RTF", "WER(%)", "Accept Rate(%)"))
    print("-" * 65)
    for name, data in all_results.items():
        wer_str = "{:.2f}".format(data["wer"] * 100) if data.get("wer") is not None else "N/A"
        accept_str = "N/A"
        if data.get("ssd_stats") and data["ssd_stats"].get("acceptance_rate") is not None:
            accept_str = "{:.1f}".format(data["ssd_stats"]["acceptance_rate"] * 100)
        print("{:<25} {:>10.4f} {:>10} {:>15}".format(name, data["mean_lm_rtf"], wer_str, accept_str))

    # Save results (without per-utterance wav results for JSON cleanliness)
    summary = {}
    for name, data in all_results.items():
        summary[name] = {k: v for k, v in data.items() if k != "results"}
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to {}".format(summary_path))


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
