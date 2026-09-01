"""
05_evaluate_asr.py
==================
Luhya / Gusii ASR — Evaluation script

Metrics:
  WER  (Word Error Rate)      — primary metric, lower is better
  CER  (Character Error Rate) — more fine-grained, useful for agglutinative langs
  eval_loss + perplexity      — from trainer_state.json (same logic as translation model)

Usage:
    python scripts/05_evaluate_asr.py --lang luhya --model-dir training/luhya/final
    python scripts/05_evaluate_asr.py --lang gusii --model-id yourname/gusii_whisper_small_v1
"""

import json
import math
import argparse
from pathlib import Path


# ── eval_loss from checkpoint (shared logic with translation model) ───────────
def parse_eval_loss(checkpoint_dir: str | None) -> dict:
    """
    Extract eval_loss and perplexity from TRL/Transformers trainer_state.json.
    Falls back to all_results.json or train_results.json.
    """
    if not checkpoint_dir:
        return {}

    cdir = Path(checkpoint_dir)

    state_path = cdir / "trainer_state.json"
    if not state_path.exists():
        for sub in sorted(cdir.glob("checkpoint-*")):
            candidate = sub / "trainer_state.json"
            if candidate.exists():
                state_path = candidate
                break

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        log_history = state.get("log_history", [])

        eval_entries = [
            {"step": e["step"], "eval_loss": e["eval_loss"]}
            for e in log_history if "eval_loss" in e
        ]
        if eval_entries:
            best  = min(eval_entries, key=lambda x: x["eval_loss"])
            final = eval_entries[-1]
            return {
                "best_eval_loss":    round(best["eval_loss"], 6),
                "final_eval_loss":   round(final["eval_loss"], 6),
                "best_step":         best["step"],
                "perplexity":        round(math.exp(best["eval_loss"]), 4),
                "eval_loss_history": eval_entries,
                "source":            str(state_path),
            }

    for fname in ("all_results.json", "train_results.json"):
        fpath = cdir / fname
        if fpath.exists():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if "eval_loss" in data:
                loss = data["eval_loss"]
                return {
                    "best_eval_loss":  round(loss, 6),
                    "final_eval_loss": round(loss, 6),
                    "perplexity":      round(math.exp(loss), 4),
                    "source":          str(fpath),
                }

    return {"eval_loss_note": f"No eval_loss found in {cdir}"}


# ── WER / CER ────────────────────────────────────────────────────────────────
def compute_wer_cer(predictions: list[str], references: list[str]) -> dict:
    import evaluate
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Normalise: lowercase + strip (consistent with training preprocessing)
    preds_norm = [p.lower().strip() for p in predictions]
    refs_norm  = [r.lower().strip() for r in references]

    wer = wer_metric.compute(predictions=preds_norm, references=refs_norm)
    cer = cer_metric.compute(predictions=preds_norm, references=refs_norm)

    return {
        "WER": round(wer * 100, 2),   # percentage
        "CER": round(cer * 100, 2),
    }


# ── main evaluation ───────────────────────────────────────────────────────────
def run_evaluation(
    model_dir_or_id: str,
    manifest_path: str,
    output_file: str,
    checkpoint_dir: str = None,
    n_samples: int = None,
    lang: str = "luhya",
):
    import torch
    import soundfile as sf
    import librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(f"\n=== {lang.upper()} ASR — Evaluation ===")
    print(f"  Model    : {model_dir_or_id}")
    print(f"  Manifest : {manifest_path}")

    # ── load model + processor ──
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    processor = WhisperProcessor.from_pretrained(model_dir_or_id, token=hf_token)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir_or_id, token=hf_token
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="sw", task="transcribe"
    )

    # ── load eval manifest ──
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if n_samples:
        import random; random.seed(42)
        records = random.sample(records, min(n_samples, len(records)))

    print(f"  Evaluating {len(records)} clips...\n")

    predictions, references, sample_output = [], [], []

    for i, record in enumerate(records):
        text_ref = record.get("text", "").strip()
        if not text_ref:
            continue

        try:
            audio, sr = sf.read(record["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception as e:
            print(f"  [skip] {record['audio_path']}: {e}")
            continue

        inputs = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=225)

        pred_text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        predictions.append(pred_text)
        references.append(text_ref)

        if i < 15:
            sample_output.append({
                "reference":  text_ref,
                "predicted":  pred_text,
                "dialect":    record.get("dialect", "?"),
                "source":     record.get("source", "?"),
                "duration_s": record.get("duration_s", 0),
            })

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(records)}")

    # ── compute metrics ──
    metrics = compute_wer_cer(predictions, references)
    eval_loss_data = parse_eval_loss(checkpoint_dir)

    results = {
        "model":     model_dir_or_id,
        "language":  lang,
        "n_clips":   len(predictions),
        **metrics,
        **eval_loss_data,
        "samples":   sample_output,
    }

    print(f"\n{'='*42}")
    print(f"  WER         : {metrics['WER']}%")
    print(f"  CER         : {metrics['CER']}%")
    if "best_eval_loss" in results:
        print(f"  eval_loss   : {results['best_eval_loss']}  (step {results.get('best_step','?')})")
        print(f"  perplexity  : {results['perplexity']}")
    print(f"{'='*42}\n")

    # ── WER interpretation ──
    wer = metrics["WER"]
    if wer < 25:
        tier = "Excellent — production-grade for controlled domains"
    elif wer < 40:
        tier = "Good — usable with post-processing / language model rescoring"
    elif wer < 55:
        tier = "Fair — useful for assisted transcription, needs more data"
    else:
        tier = "Needs improvement — gather more audio or review pseudo-labels"
    print(f"  Assessment  : {tier}\n")

    print("Sample predictions:")
    for s in sample_output[:5]:
        print(f"  REF : {s['reference']}")
        print(f"  PRED: {s['predicted']}  [{s['dialect']}]")
        print()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {output_file}")
    return results


# ── entry point ───────────────────────────────────────────────────────────────
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",            required=True, choices=["luhya", "gusii"])
    parser.add_argument("--model-dir",       default=None)
    parser.add_argument("--model-id",        default=None)
    parser.add_argument("--manifest",        default=None,
                        help="Override default manifest path")
    parser.add_argument("--checkpoint-dir",  default=None,
                        help="Training checkpoint dir for eval_loss parsing")
    parser.add_argument("--output",          default=None)
    parser.add_argument("--n-samples",       type=int, default=None)
    args = parser.parse_args()

    base = Path(__file__).parent.parent

    model_ref = args.model_dir or args.model_id
    if not model_ref:
        raise ValueError("Provide --model-dir or --model-id")

    manifest = args.manifest or str(
        base / f"data/processed/{args.lang}/manifest_eval.jsonl"
    )
    output   = args.output or str(
        base / f"evaluation/{args.lang}_results.json"
    )
    ckpt_dir = args.checkpoint_dir or str(
        base / f"training/{args.lang}/checkpoints"
    )

    run_evaluation(
        model_dir_or_id=model_ref,
        manifest_path=manifest,
        output_file=output,
        checkpoint_dir=ckpt_dir,
        n_samples=args.n_samples,
        lang=args.lang,
    )
