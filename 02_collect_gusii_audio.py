"""
02_collect_gusii_audio.py
=========================
Gusii/Ekegusii ASR — Data collection pipeline

No public Gusii speech corpus exists. This script implements three
collection strategies meant to be run in combination:

  A. YouTube/radio download (yt-dlp) — Egesa FM, KBC Gusii, Citizen TV clips
  B. Bible TTS bootstrap — Faith Comes By Hearing audio Bible alignment
  C. Contributed recordings — your own speaker recordings

For A and B, transcripts don't come with the audio — we use:
  - Whisper large-v3 to produce a rough transcript (pseudo-label)
  - Manual correction by a native Gusii speaker (recorded in correction_log.jsonl)

Output mirrors 01_prepare_luhya_audio.py:
  data/processed/gusii/manifest_train.jsonl
  data/processed/gusii/manifest_eval.jsonl
  data/processed/gusii/stats.json
  data/raw/gusii/pseudo_labels/   ← Whisper-generated, needs human review
  data/raw/gusii/correction_log.jsonl  ← human corrections

Recommended target before training:
  - Minimum: 10 hrs (expect WER ~45–55% with whisper-small)
  - Good:    20 hrs (expect WER ~30–40%)
  - Strong:  30+ hrs (expect WER ~22–32%)
"""

import json
import subprocess
import argparse
import shutil
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

SEED = 42
EVAL_FRACTION  = 0.10
MIN_DURATION_S = 1.0
MAX_DURATION_S = 30.0
TARGET_SR      = 16000

# ── known Gusii audio sources ─────────────────────────────────────────────────
# These are starting points — add more as you find them.
# Egesa FM broadcasts almost entirely in Ekegusii.
YOUTUBE_SOURCES = [
    # (url_or_channel, description)
    ("https://www.youtube.com/@EgesaFM",          "Egesa FM — Gusii language radio"),
    ("https://www.youtube.com/@KBCChannel1Kenya",  "KBC — occasional Gusii segments"),
    # Add specific playlist/video URLs here as you find them:
    # ("https://www.youtube.com/watch?v=XXXX", "Egesa FM news bulletin"),
]

# Bible TTS: Faith Comes By Hearing records audio Bibles in African languages.
# Ekegusii New Testament: request at https://www.faithcomesbyhearing.com/audio-bible-resources/
# When downloaded, it arrives as MP3 files named by book+chapter (e.g. GEN_001.mp3)
BIBLE_CHAPTERS_DIR = "data/raw/gusii/bible_audio"     # place MP3s here
BIBLE_TRANSCRIPT_DIR = "data/raw/gusii/bible_text"    # plain text NT by verse


@dataclass
class AudioRecord:
    audio_path: str
    text: str
    source: str
    duration_s: float
    pseudo_labeled: bool = False   # True = Whisper transcript, needs human review
    human_verified: bool = False   # True = corrected by native speaker
    speaker_id: str = "unknown"


# ── utilities ─────────────────────────────────────────────────────────────────
def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def get_duration(wav_path: Path) -> float:
    try:
        import soundfile as sf
        return sf.info(str(wav_path)).duration
    except Exception:
        size = wav_path.stat().st_size - 44
        return max(0.0, size / (TARGET_SR * 2))


def to_wav(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = run([
        "ffmpeg", "-y", "-i", str(src),
        "-ar", str(TARGET_SR), "-ac", "1", "-sample_fmt", "s16",
        str(dst),
    ])
    return result.returncode == 0


def chunk_audio(wav_path: Path, chunk_dir: Path, chunk_s: float = 25.0) -> list[Path]:
    """
    Split a long WAV into ~25s chunks using ffmpeg segment.
    Whisper max context = 30s; 25s leaves headroom.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / f"{wav_path.stem}_%04d.wav")
    run([
        "ffmpeg", "-y", "-i", str(wav_path),
        "-f", "segment",
        "-segment_time", str(chunk_s),
        "-ar", str(TARGET_SR), "-ac", "1", "-sample_fmt", "s16",
        "-reset_timestamps", "1",
        pattern,
    ])
    return sorted(chunk_dir.glob(f"{wav_path.stem}_*.wav"))


# ── A. YouTube / radio download ───────────────────────────────────────────────
def download_youtube(urls: list[str], out_dir: Path, max_per_source: int = 50) -> list[Path]:
    """
    Download audio from YouTube channels/playlists using yt-dlp.
    Converts directly to WAV 16kHz mono.
    Returns list of downloaded WAV paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for url in urls:
        print(f"  [yt-dlp] Downloading from: {url}")
        result = run([
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"-ar {TARGET_SR} -ac 1",
            "--output", str(out_dir / "%(id)s.%(ext)s"),
            "--playlist-end", str(max_per_source),
            "--no-overwrites",
            "--ignore-errors",
            url,
        ])
        if result.returncode != 0:
            print(f"    [warn] yt-dlp error: {result.stderr[:200]}")

        downloaded += sorted(out_dir.glob("*.wav"))

    print(f"  [youtube] {len(downloaded)} audio files downloaded")
    return downloaded


# ── B. Bible TTS alignment ───────────────────────────────────────────────────
def align_bible(mp3_dir: Path, text_dir: Path, out_dir: Path) -> list[AudioRecord]:
    """
    Align Ekegusii Bible audio (MP3, by chapter) to verse-level text.

    Strategy:
      1. Convert each chapter MP3 → WAV
      2. Use Whisper to transcribe (pseudo-label) if no aligned text
      3. If verse-level text exists, use forced alignment via whisperx
         to get per-verse timestamps → chunk at verse boundaries
      4. Chunks >30s are split further at silence boundaries

    Returns AudioRecord list. All records are pseudo_labeled=True unless
    whisperx forced alignment succeeded.
    """
    records: list[AudioRecord] = []
    if not mp3_dir.exists():
        print(f"  [bible] Directory not found: {mp3_dir}")
        print("  Request Ekegusii NT audio from https://www.faithcomesbyhearing.com")
        return records

    wav_dir   = out_dir / "bible_wav"
    chunk_dir = out_dir / "bible_chunks"

    for mp3_path in sorted(mp3_dir.glob("*.mp3")):
        wav_path = wav_dir / (mp3_path.stem + ".wav")
        if not wav_path.exists():
            if not to_wav(mp3_path, wav_path):
                continue

        # Check for parallel text file (same stem)
        txt_path = text_dir / (mp3_path.stem + ".txt")

        # Try whisperx forced alignment first (gives word-level timestamps)
        aligned = False
        if txt_path.exists():
            try:
                import whisperx   # pip install whisperx
                result = run([
                    "whisperx", str(wav_path),
                    "--language", "sw",     # closest Bantu proxy
                    "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
                    "--output_dir", str(chunk_dir / mp3_path.stem),
                    "--output_format", "json",
                ])
                if result.returncode == 0:
                    aligned = True
            except (ImportError, FileNotFoundError):
                pass   # whisperx not installed — fall through to chunking

        if not aligned:
            # Naive chunking: split into 25s segments, pseudo-label each
            chunks = chunk_audio(wav_path, chunk_dir / mp3_path.stem)
            for chunk in chunks:
                dur = get_duration(chunk)
                if not (MIN_DURATION_S <= dur <= MAX_DURATION_S):
                    continue
                records.append(AudioRecord(
                    audio_path=str(chunk),
                    text="",               # filled by pseudo-labelling step
                    source="bible_tts",
                    duration_s=round(dur, 3),
                    pseudo_labeled=True,
                    human_verified=False,
                ))

    print(f"  [bible] {len(records)} chunks prepared from Bible audio")
    return records


# ── C. Pseudo-labelling with Whisper ─────────────────────────────────────────
def pseudo_label(
    records: list[AudioRecord],
    out_dir: Path,
    whisper_model: str = "openai/whisper-large-v3",
    batch_size: int = 8,
) -> list[AudioRecord]:
    """
    Run Whisper large-v3 over unlabelled clips to produce draft transcripts.
    These MUST be reviewed by a native Gusii speaker before training.

    Whisper has no Ekegusii token. We force decode with language="sw" (Swahili)
    as the closest supported Bantu language — output will be noisy but usable
    as a starting point for human correction.
    """
    unlabelled = [r for r in records if not r.text and r.pseudo_labeled]
    if not unlabelled:
        print("  [pseudo-label] Nothing to label")
        return records

    print(f"  [pseudo-label] Labelling {len(unlabelled)} clips with {whisper_model}...")

    try:
        import torch
        from transformers import pipeline as hf_pipeline

        asr = hf_pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            generate_kwargs={"language": "sw", "task": "transcribe"},
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,
        )

        log_path = out_dir / "pseudo_labels" / "labels.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as log:
            for i in range(0, len(unlabelled), batch_size):
                batch = unlabelled[i:i+batch_size]
                audio_paths = [r.audio_path for r in batch]
                try:
                    results = asr(audio_paths, batch_size=batch_size)
                    for r, res in zip(batch, results):
                        r.text = res["text"].strip()
                        log.write(json.dumps({
                            "audio": r.audio_path,
                            "pseudo_text": r.text,
                            "human_corrected": "",   # fill this in
                            "verified": False,
                        }, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"    [pseudo-label] batch {i//batch_size} error: {e}")

                if (i + batch_size) % 50 == 0:
                    print(f"    ... {i+batch_size}/{len(unlabelled)} done")

        print(f"  [pseudo-label] Done. Review → {log_path}")
        print("  ⚠  These transcripts need native Gusii speaker review before training!")

    except ImportError:
        print("  [pseudo-label] transformers not installed — skipping")

    return records


# ── D. Load human corrections ─────────────────────────────────────────────────
def apply_corrections(records: list[AudioRecord], correction_log: Path) -> list[AudioRecord]:
    """
    Merge human corrections from correction_log.jsonl into records.
    Format of each line:
      {"audio": "/path/to/clip.wav", "human_corrected": "Corrected text here", "verified": true}
    """
    if not correction_log.exists():
        return records

    corrections: dict[str, str] = {}
    with correction_log.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("verified") and obj.get("human_corrected"):
                corrections[obj["audio"]] = obj["human_corrected"]

    for r in records:
        if r.audio_path in corrections:
            r.text           = corrections[r.audio_path]
            r.human_verified = True

    n_corrected = sum(1 for r in records if r.human_verified)
    print(f"  [corrections] Applied {n_corrected} human corrections")
    return records


# ── load contributed recordings ───────────────────────────────────────────────
def load_contributed(contributed_dir: Path, out_dir: Path) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    if not contributed_dir.exists():
        return records

    for wav_path in sorted(contributed_dir.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        dur = get_duration(wav_path)
        if not (MIN_DURATION_S <= dur <= MAX_DURATION_S):
            continue
        dst = out_dir / "audio" / "contributed" / wav_path.name
        if not dst.exists():
            to_wav(wav_path, dst)
        records.append(AudioRecord(
            audio_path=str(dst),
            text=text,
            source="contributed",
            duration_s=round(dur, 3),
            human_verified=True,
            speaker_id=wav_path.stem.split("_")[0],
        ))

    print(f"  [contributed] {len(records)} clips loaded")
    return records


# ── split + write ─────────────────────────────────────────────────────────────
def make_split(records: list[AudioRecord]) -> tuple[list, list]:
    # Only use clips that have a text (pseudo or verified)
    usable = [r for r in records if r.text and len(r.text) > 2]
    if len(usable) < len(records):
        print(f"  [split] {len(records)-len(usable)} clips have no text yet — excluded")

    # Speaker-aware split (same logic as Luhya pipeline)
    by_speaker: dict[str, list] = defaultdict(list)
    for r in usable:
        by_speaker[r.speaker_id].append(r)

    random.seed(SEED)
    speakers = list(by_speaker.keys())
    random.shuffle(speakers)
    n_eval = max(1, int(len(speakers) * EVAL_FRACTION))

    eval_spk  = set(speakers[:n_eval])
    train_spk = set(speakers[n_eval:])

    train = [r for r in usable if r.speaker_id in train_spk]
    eval_ = [r for r in usable if r.speaker_id in eval_spk]
    return train, eval_


def write_manifest(records: list[AudioRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",      default="data/processed/gusii")
    parser.add_argument("--contributed-dir", default="data/raw/gusii/contributed")
    parser.add_argument("--bible-audio",     default=BIBLE_CHAPTERS_DIR)
    parser.add_argument("--bible-text",      default=BIBLE_TRANSCRIPT_DIR)
    parser.add_argument("--youtube",         action="store_true", help="Download from YouTube sources")
    parser.add_argument("--max-yt-per-src",  type=int, default=50)
    parser.add_argument("--pseudo-label",    action="store_true", help="Run Whisper pseudo-labelling")
    parser.add_argument("--correction-log",  default="data/raw/gusii/correction_log.jsonl")
    parser.add_argument("--whisper-model",   default="openai/whisper-large-v3")
    args = parser.parse_args()

    base        = Path(__file__).parent.parent
    out_dir     = base / args.output_dir
    raw_dir     = base / "data/raw/gusii"

    print("\n=== Gusii ASR — Data Collection Pipeline ===\n")
    all_records: list[AudioRecord] = []

    # A. YouTube
    if args.youtube:
        print("Downloading YouTube sources...")
        yt_wavs = download_youtube(
            [url for url, _ in YOUTUBE_SOURCES],
            raw_dir / "youtube",
            max_per_source=args.max_yt_per_src,
        )
        for wav in yt_wavs:
            dur = get_duration(wav)
            if dur > MAX_DURATION_S:
                chunks = chunk_audio(wav, raw_dir / "youtube_chunks")
                for c in chunks:
                    all_records.append(AudioRecord(
                        audio_path=str(c),
                        text="",
                        source="youtube",
                        duration_s=round(get_duration(c), 3),
                        pseudo_labeled=True,
                    ))
            elif dur >= MIN_DURATION_S:
                all_records.append(AudioRecord(
                    audio_path=str(wav),
                    text="",
                    source="youtube",
                    duration_s=round(dur, 3),
                    pseudo_labeled=True,
                ))

    # B. Bible TTS
    print("Loading Bible TTS audio...")
    all_records += align_bible(
        base / args.bible_audio,
        base / args.bible_text,
        raw_dir,
    )

    # C. Contributed recordings
    print("Loading contributed recordings...")
    all_records += load_contributed(base / args.contributed_dir, out_dir)

    # D. Pseudo-labelling
    if args.pseudo_label:
        print("Running Whisper pseudo-labelling...")
        all_records = pseudo_label(all_records, raw_dir, whisper_model=args.whisper_model)

    # E. Apply human corrections
    print("Applying human corrections...")
    all_records = apply_corrections(all_records, base / args.correction_log)

    # Split + write
    train, eval_ = make_split(all_records)
    write_manifest(train, out_dir / "manifest_train.jsonl")
    write_manifest(eval_,  out_dir / "manifest_eval.jsonl")

    total_hrs = sum(r.duration_s for r in train + eval_) / 3600
    verified  = sum(1 for r in train + eval_ if r.human_verified)

    stats = {
        "total_clips":    len(train) + len(eval_),
        "total_hours":    round(total_hrs, 2),
        "train_clips":    len(train),
        "eval_clips":     len(eval_),
        "human_verified": verified,
        "pseudo_only":    len(train) + len(eval_) - verified,
        "sources":        list({r.source for r in train + eval_}),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n✓ Done!")
    print(f"  Total : {stats['total_clips']} clips | {stats['total_hours']} hrs")
    print(f"  Verified by human : {verified}")
    print(f"  Pseudo-labelled (needs review): {stats['pseudo_only']}")
    if total_hrs < 10:
        print(f"\n  ⚠  {total_hrs:.1f}hrs is below the 10hr minimum for usable ASR.")
        print("     Recruit native Gusii speakers — even 5 speakers × 2hrs = 10hrs.")


if __name__ == "__main__":
    main()
