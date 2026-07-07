"""
01_prepare_data.py
==================
Luhya TranslateGemma — Data preparation pipeline

Sources handled:
  1. KenTrans (Harvard Dataverse) — Luhya-Swahili pairs, O:/T: format
     https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NOAT0W
  2. Bible corpus (Luhya-Marama / English NT) — plain parallel text files
  3. Your own contributed pairs (CSV/JSONL)

Strategy:
  - KenTrans pairs are Luhya→Swahili. We pivot through NLLB-200 (Helsinki or
    facebook/nllb-200-distilled-600M) to get Swahili→English, giving us
    Luhya↔English pairs. Quality-filtered at chrF >= 0.35.
  - Bible corpus is already Luhya↔English; loaded directly.
  - Contributed pairs are loaded as-is (trusted source).
  - All merged, deduplicated, shuffled, and split 95/5 train/eval.

Output:
  data/processed/train.jsonl
  data/processed/eval.jsonl
  data/processed/stats.json
"""

import json
import re
import csv
import hashlib
import random
import argparse
from pathlib import Path
from typing import Iterator

# ── constants ───────────────────────────────────────────────────────────────
SEED = 42
EVAL_FRACTION = 0.05

# Dialects we care about (KenTrans file name substrings → dialect tag)
DIALECT_MAP = {
    "bukusu": "lubukusu",
    "tachoni": "lutachobi",
    "marachi": "lumarachi",
    "maragoli": "lulogooli",
    "banyore": "lunyole",
    "kisa": "oluShisa",
    "marama": "oluMarama",
    "tsotso": "oluTsotso",
    "wanga": "luwanga",
    "nyala":"lunyala",
    "idakho": "luidakho",
    "isukha": "luisukha",
    "tiriki": "ludiriji",
    "samia": "lusamia",
    "khayo": "lukhayo",
    "kabras": "lukabras", 
}

SUPPORTED_DIALECTS = {"lumarachi", "lulogooli"}   # exclude lubukusu for this run


# ── helpers ─────────────────────────────────────────────────────────────────
def pair_hash(en: str, luhya: str) -> str:
    """Stable dedup key."""
    return hashlib.md5(f"{en.strip()}|||{luhya.strip()}".encode()).hexdigest()


def is_junk(text: str) -> bool:
    """Return True if sentence looks like corpus noise."""
    text = text.strip()
    if len(text) < 5 or len(text) > 400:
        return True
    # mostly digits / punctuation
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.40:
        return True
    return False


def detect_dialect(filename: str) -> str | None:
    """Guess dialect from filename, return None if not in supported set."""
    fname = filename.lower()
    for key, tag in DIALECT_MAP.items():
        if key in fname:
            return tag if tag in SUPPORTED_DIALECTS else None
    return None


# ── 1. KenTrans O:/T: parser ────────────────────────────────────────────────
def parse_kentrans_file(path: Path, dialect: str) -> list[dict]:
    """
    KenTrans format:
        O: <luhya sentence>
        T: <swahili sentence>
        (blank line)
    Returns list of {"luhya": ..., "swahili": ..., "dialect": ..., "source": "kentrans"}
    """
    pairs = []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        luhya_lines, swahili_lines = [], []
        for line in lines:
            if line.startswith("O:"):
                luhya_lines.append(line[2:].strip())
            elif line.startswith("T:"):
                swahili_lines.append(line[2:].strip())
        luhya = " ".join(luhya_lines).strip()
        swahili = " ".join(swahili_lines).strip()
        if luhya and swahili and not is_junk(luhya) and not is_junk(swahili):
            pairs.append({
                "luhya": luhya,
                "swahili": swahili,
                "dialect": dialect,
                "source": "kentrans",
            })
    return pairs


def load_all_kentrans(kentrans_dir: Path) -> list[dict]:
    """Walk directory and load all matching KenTrans files."""
    all_pairs = []
    for f in sorted(kentrans_dir.rglob("*.txt")):
        dialect = detect_dialect(f.name)
        if dialect is None:
            print(f"  [skip] {f.name} — not a supported dialect")
            continue
        pairs = parse_kentrans_file(f, dialect)
        print(f"  [kentrans] {f.name} → {len(pairs)} pairs ({dialect})")
        all_pairs.extend(pairs)
    return all_pairs


# ── 2. Bible corpus (plain parallel files) ─────────────────────────────────
def load_bible_corpus(src_file: Path, tgt_file: Path, dialect: str = "lumarachi") -> list[dict]:
    """
    Expects two aligned plain-text files, one sentence per line:
      src_file: Luhya (e.g. Marama NT)
      tgt_file: English (e.g. KJV or NIV NT)
    """
    src_lines = src_file.read_text(encoding="utf-8", errors="replace").splitlines()
    tgt_lines = tgt_file.read_text(encoding="utf-8", errors="replace").splitlines()

    pairs = []
    for luhya, english in zip(src_lines, tgt_lines):
        luhya, english = luhya.strip(), english.strip()
        if luhya and english and not is_junk(luhya) and not is_junk(english):
            pairs.append({
                "luhya": luhya,
                "english": english,
                "dialect": dialect,
                "source": "bible",
            })
    print(f"  [bible] {src_file.name} / {tgt_file.name} → {len(pairs)} pairs")
    return pairs


# ── 3. Pivot Luhya-Swahili → English via NLLB ──────────────────────────────
# Confidence scoring strategy
# ─────────────────────────────
# NLLB sequence scores are log-probabilities summed over output tokens.
# A raw sum is length-biased (longer outputs score lower), so we
# length-normalise: score_per_tok = total_log_prob / n_output_tokens.
# We then convert to a [0, 1] confidence via exp(score_per_tok).
# Pairs below PIVOT_CONF_THRESHOLD are discarded before they ever reach
# the training set; pairs above PIVOT_CONF_WARN are flagged in the stats.
#
# Empirically for NLLB-600M on Swahili→English:
#   exp(score_per_tok) ≈ 0.55–0.75  → fluent, high-confidence
#   exp(score_per_tok) ≈ 0.35–0.55  → acceptable, keep with caution
#   exp(score_per_tok) < 0.35       → likely noise or hallucination, discard

PIVOT_CONF_THRESHOLD = 0.35   # hard reject below this
PIVOT_CONF_WARN      = 0.45   # warn-but-keep band: 0.35–0.45


def _normalised_confidence(sequences_score: float, n_tokens: int) -> float:
    """Convert a Hugging Face generate() sequences_score to a [0,1] confidence."""
    import math
    if n_tokens == 0:
        return 0.0
    score_per_tok = sequences_score / n_tokens   # length-normalised log-prob
    return math.exp(score_per_tok)               # map to (0, 1]


def pivot_swahili_to_english(
    pairs: list[dict],
    batch_size: int = 16,
    conf_threshold: float = PIVOT_CONF_THRESHOLD,
) -> list[dict]:
    """
    Use facebook/nllb-200-distilled-600M to translate the Swahili side
    to English, producing Luhya↔English pairs.

    Confidence filtering
    --------------------
    Each translated pair receives a ``pivot_confidence`` score in [0, 1]
    derived from the model's own sequence log-probability (length-normalised).
    Pairs below ``conf_threshold`` (default 0.35) are dropped. The threshold
    can be overridden via --pivot-conf-threshold on the CLI.

    Only runs if `transformers` is available; falls back gracefully.
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("  [pivot] transformers not installed — skipping pivot")
        for p in pairs:
            p["english"] = None
            p["pivot_confidence"] = 0.0
        return pairs

    model_name = "facebook/nllb-200-distilled-600M"
    print(f"  [pivot] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nllb_model.to(device).eval()
    print(f"  [pivot] Running on {device} | conf_threshold={conf_threshold}")

    forced_bos_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    output_pairs: list[dict] = []
    n_kept = n_low_conf = n_junk = 0

    for batch_idx, batch in enumerate(batches):
        swahili_texts = [p["swahili"] for p in batch]
        try:
            enc = tokenizer(
                swahili_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                src_lang="swh_Latn",
            ).to(device)

            with torch.no_grad():
                out = nllb_model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos_id,
                    max_new_tokens=256,
                    num_beams=4,
                    early_stopping=True,
                    # Return sequence scores so we can compute confidence
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=False,
                )

            # sequences_scores is the beam-search final score (sum of log-probs)
            seq_scores = out.sequences_scores.tolist()
            decoded    = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
            n_tokens   = [(out.sequences[i] != tokenizer.pad_token_id).sum().item()
                          for i in range(len(decoded))]

            for p, translated, seq_score, n_tok in zip(batch, decoded, seq_scores, n_tokens):
                translated = translated.strip()

                if is_junk(translated):
                    n_junk += 1
                    continue

                confidence = _normalised_confidence(seq_score, n_tok)

                if confidence < conf_threshold:
                    n_low_conf += 1
                    continue

                new_p = dict(p)
                new_p["english"]          = translated
                new_p["pivot_confidence"] = round(confidence, 4)
                new_p["pivot_quality"]    = confidence   # kept for build_final_dataset compat
                if confidence < PIVOT_CONF_WARN:
                    new_p["pivot_warn"] = True           # visible in stats
                output_pairs.append(new_p)
                n_kept += 1

        except Exception as e:
            print(f"    [pivot] batch {batch_idx} error: {e}")

        if (batch_idx + 1) % 10 == 0:
            done = (batch_idx + 1) * batch_size
            print(f"    ... {done}/{len(pairs)} | kept={n_kept} low_conf={n_low_conf} junk={n_junk}")

    # Summary histogram of confidence bands
    bands = {"[0.35,0.45)": 0, "[0.45,0.55)": 0, "[0.55,0.65)": 0, "[0.65,1.0]": 0}
    for p in output_pairs:
        c = p["pivot_confidence"]
        if c < 0.45:   bands["[0.35,0.45)"] += 1
        elif c < 0.55: bands["[0.45,0.55)"] += 1
        elif c < 0.65: bands["[0.55,0.65)"] += 1
        else:          bands["[0.65,1.0]"]  += 1

    print(
        f"  [pivot] done | input={len(pairs)} kept={n_kept} "
        f"low_conf={n_low_conf} junk={n_junk}"
    )
    print(f"  [pivot] confidence distribution: {bands}")
    return output_pairs


# ── 4. Contributed pairs (CSV or JSONL) ────────────────────────────────────
def load_contributed_pairs(contributed_dir: Path) -> list[dict]:
    """
    Load user-contributed pairs from:
      - CSV with columns: english, luhya, dialect (optional), notes (optional)
      - JSONL with keys:  english, luhya, dialect (optional)
    """
    all_pairs = []

    for f in sorted(contributed_dir.rglob("*.csv")):
        with f.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                en = row.get("english", "").strip()
                lh = row.get("luhya", "").strip()
                if en and lh:
                    all_pairs.append({
                        "english": en,
                        "luhya": lh,
                        "dialect": row.get("dialect", "unknown").strip() or "unknown",
                        "source": "contributed",
                        "notes": row.get("notes", ""),
                    })
        print(f"  [contributed] {f.name} → {len(all_pairs)} pairs")

    for f in sorted(contributed_dir.rglob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                en = obj.get("english", "").strip()
                lh = obj.get("luhya", "").strip()
                if en and lh:
                    all_pairs.append({
                        "english": en,
                        "luhya": lh,
                        "dialect": obj.get("dialect", "unknown"),
                        "source": "contributed",
                    })
        print(f"  [contributed] {f.name} loaded")

    return all_pairs


# ── 5. Dedup, merge, split ──────────────────────────────────────────────────
def build_final_dataset(
    bible_pairs: list[dict],
    pivoted_pairs: list[dict],
    contributed_pairs: list[dict],
) -> tuple[list[dict], list[dict]]:

    seen = set()
    merged = []

    def add(record: dict):
        en = record.get("english") or ""
        lh = record.get("luhya") or ""
        if not en or not lh:
            return
        key = pair_hash(en, lh)
        if key in seen:
            return
        seen.add(key)
        merged.append({
            "english": en.strip(),
            "luhya": lh.strip(),
            "dialect": record.get("dialect", "unknown"),
            "source": record.get("source", "unknown"),
        })

    # Priority: contributed > bible > pivoted (lower quality)
    for p in contributed_pairs:
        add(p)
    for p in bible_pairs:
        add(p)
    for p in pivoted_pairs:
        if p.get("pivot_quality", 0) >= 0.5:
            add(p)

    random.seed(SEED)
    random.shuffle(merged)

    n_eval = max(50, int(len(merged) * EVAL_FRACTION))
    eval_set  = merged[:n_eval]
    train_set = merged[n_eval:]

    return train_set, eval_set


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Luhya data pipeline")
    parser.add_argument("--kentrans-dir",    default="data/raw/kentrans",    help="Directory with KenTrans .txt files")
    parser.add_argument("--bible-luhya",     default="data/raw/bible/luhya_nt.txt",  help="Luhya Bible NT (one sentence per line)")
    parser.add_argument("--bible-english",   default="data/raw/bible/english_nt.txt",help="English Bible NT (aligned)")
    parser.add_argument("--contributed-dir", default="data/contributed",     help="Your own CSV/JSONL pairs")
    parser.add_argument("--output-dir",      default="data/processed",       help="Output directory")
    parser.add_argument("--skip-pivot",      action="store_true",            help="Skip NLLB pivot (faster; Bible+contributed only)")
    parser.add_argument("--pivot-conf-threshold", type=float, default=PIVOT_CONF_THRESHOLD,
                        help=f"Min confidence to keep a pivoted pair (default {PIVOT_CONF_THRESHOLD}). "
                             f"Range [0,1]; lower = keep more pairs but noisier.")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    kentrans_dir    = base / args.kentrans_dir
    bible_luhya     = base / args.bible_luhya
    bible_english   = base / args.bible_english
    contributed_dir = base / args.contributed_dir
    output_dir      = base / args.output_dir

    print("\n=== Luhya TranslateGemma — Data Pipeline ===\n")

    # 1. KenTrans
    kentrans_pairs = []
    if kentrans_dir.exists():
        print("Loading KenTrans pairs...")
        kentrans_pairs = load_all_kentrans(kentrans_dir)
        print(f"  → {len(kentrans_pairs)} total Luhya-Swahili pairs\n")
    else:
        print(f"  [warn] KenTrans dir not found: {kentrans_dir}\n")

    # 2. Bible
    bible_pairs = []
    if bible_luhya.exists() and bible_english.exists():
        print("Loading Bible corpus...")
        bible_pairs = load_bible_corpus(bible_luhya, bible_english)
        print()
    else:
        print(f"  [warn] Bible files not found. Expected:\n    {bible_luhya}\n    {bible_english}\n")

    # 3. Contributed
    contributed_pairs = []
    if contributed_dir.exists():
        print("Loading contributed pairs...")
        contributed_pairs = load_contributed_pairs(contributed_dir)
        print(f"  → {len(contributed_pairs)} contributed pairs\n")

    # 4. Pivot
    pivoted_pairs = []
    if kentrans_pairs and not args.skip_pivot:
        print("Pivoting Luhya-Swahili → English via NLLB-200...")
        pivoted_pairs = pivot_swahili_to_english(
            kentrans_pairs,
            conf_threshold=args.pivot_conf_threshold,
        )
        print()
    elif args.skip_pivot:
        print("[skip-pivot] Skipping NLLB pivot as requested.\n")

    # 5. Merge + split
    print("Building final dataset...")
    train_set, eval_set = build_final_dataset(bible_pairs, pivoted_pairs, contributed_pairs)

    write_jsonl(train_set, output_dir / "train.jsonl")
    write_jsonl(eval_set,  output_dir / "eval.jsonl")

    # Dialect breakdown
    from collections import Counter
    dialect_counts = Counter(r["dialect"] for r in train_set + eval_set)
    source_counts  = Counter(r["source"]  for r in train_set + eval_set)

    stats = {
        "total_pairs": len(train_set) + len(eval_set),
        "train_pairs": len(train_set),
        "eval_pairs":  len(eval_set),
        "dialect_distribution": dict(dialect_counts),
        "source_distribution":  dict(source_counts),
        "pivot_conf_threshold": args.pivot_conf_threshold,
        "pivot_low_conf_warned": sum(
            1 for r in train_set + eval_set if r.get("pivot_warn")
        ),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n✓ Done!")
    print(f"  Train : {len(train_set):,} pairs → {output_dir}/train.jsonl")
    print(f"  Eval  : {len(eval_set):,}  pairs → {output_dir}/eval.jsonl")
    print(f"  Dialect breakdown: {dict(dialect_counts)}")
    print(f"  Source breakdown:  {dict(source_counts)}")


if __name__ == "__main__":
    main()
