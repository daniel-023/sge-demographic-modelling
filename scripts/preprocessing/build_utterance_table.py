#!/usr/bin/env python3

"""
Build per-part utterance tables from speaker directories and split files.

This script supports both layouts:
1) Merged: SPEAKERxxxx/wav/*.wav
2) Unmerged fallback: SPEAKERxxxx/SESSION*/**/*.wav
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import os
from typing import Optional

import pandas as pd
import soundfile as sf
from tqdm import tqdm


SPEAKER_ID_RE = re.compile(r"(\d{4,10}(?:-\d+)?)$")
PART3_CONF_SPEAKER_ID_RE = re.compile(r"(\d{6,10})(?:\D.*)?$")


def load_split(split_dir: Path, part: int, split: str) -> set[str]:
    path = split_dir / f"part{part}_{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_split_map(split_dir: Path, part: int) -> dict[str, str]:
    train_ids = load_split(split_dir, part, "train")
    val_ids = load_split(split_dir, part, "val")
    test_ids = load_split(split_dir, part, "test")

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"Split overlap detected for part {part}: {len(overlap)} speaker IDs")

    split_map: dict[str, str] = {}
    for speaker_id in train_ids:
        split_map[speaker_id] = "train"
    for speaker_id in val_ids:
        split_map[speaker_id] = "val"
    for speaker_id in test_ids:
        split_map[speaker_id] = "test"
    return split_map


def find_speakers_dir(audio_root_base: Path, part: int) -> Path:
    candidates = [
        audio_root_base / f"part_{part}" / "speakers",
        audio_root_base / f"part{part}" / "speakers",
        audio_root_base / f"part_{part}",
        audio_root_base / f"part{part}",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find speakers dir for part {part}. Tried: {tried}")


def resolve_speaker_id(folder_name: str, part: int, valid_speaker_ids: set[str]) -> Optional[str]:
    candidates: list[str] = []
    raw = folder_name.strip()
    candidates.append(raw)

    match = SPEAKER_ID_RE.search(raw)
    if match:
        sid = match.group(1)
        candidates.append(sid)
        if sid.isdigit() and len(sid) <= 4:
            candidates.append(sid.zfill(4))

    for c in list(candidates):
        candidates.append(f"part{part}_{c}")
        candidates.append(f"part_{part}_{c}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in valid_speaker_ids:
            return candidate
    return None


def iter_wavs(spk_dir: Path):
    wav_dir = spk_dir / "wav"
    if wav_dir.is_dir():
        for path in sorted(wav_dir.iterdir()):
            if path.is_file() and path.suffix.lower() == ".wav":
                yield path
        return

    found_in_sessions = False
    for child in sorted(spk_dir.iterdir()):
        if child.is_dir() and child.name.upper().startswith("SESSION"):
            for path in sorted(child.iterdir()):
                if path.is_file() and path.suffix.lower() == ".wav":
                    found_in_sessions = True
                    yield path

    if found_in_sessions:
        return

    for path in sorted(spk_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".wav":
            yield path


def wav_duration_sec(path: Path) -> Optional[float]:
    try:
        with sf.SoundFile(path) as audio:
            if audio.samplerate <= 0:
                return None
            return len(audio) / audio.samplerate
    except Exception:
        return None


def collect_part_wavs(part: int, audio_root_base: Path, split_dir: Path):
    split_map = build_split_map(split_dir, part)

    speakers_dir = find_speakers_dir(audio_root_base, part)
    print(f"Part {part}: speakers dir = {speakers_dir}")
    print(
        f"Part {part}: split sizes -> "
        f"train={sum(1 for v in split_map.values() if v == 'train')}, "
        f"val={sum(1 for v in split_map.values() if v == 'val')}, "
        f"test={sum(1 for v in split_map.values() if v == 'test')}"
    )

    wav_metadata = []
    matched_speakers = 0
    unmatched_dirs = 0

    for spk_dir in sorted(path for path in speakers_dir.iterdir() if path.is_dir()):
        speaker_id = resolve_speaker_id(spk_dir.name, part, set(split_map))
        if speaker_id is None:
            unmatched_dirs += 1
            continue

        matched_speakers += 1
        split = split_map[speaker_id]
        for wav in iter_wavs(spk_dir):
            wav_metadata.append((part, wav, speaker_id, split))

    print(
        f"Part {part}: matched speakers={matched_speakers}, unmatched dirs={unmatched_dirs}, "
        f"wav files discovered={len(wav_metadata)}"
    )
    return wav_metadata


def parse_part3_speaker_id_from_stem(stem: str) -> Optional[str]:
    match = PART3_CONF_SPEAKER_ID_RE.search(stem)
    if not match:
        return None
    speaker_id = match.group(1).lstrip("0")
    return speaker_id or "0"


def canonicalize_part3_stem(stem: str) -> str:
    canonical = stem
    canonical = re.sub(r"_edited.*$", "", canonical, flags=re.IGNORECASE)
    canonical = re.sub(r"__\d+_?$", "", canonical)
    canonical = re.sub(r"-\d+$", "", canonical)
    return canonical


def index_part3_wavs(audio_dir: Path):
    if not audio_dir.exists():
        raise FileNotFoundError(f"Part 3 audio dir not found: {audio_dir}")

    wav_paths = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
        and "sliced_utterances" not in path.parts
    )

    by_exact_stem: dict[str, Path] = {}
    by_canonical_stem: dict[str, list[Path]] = defaultdict(list)
    by_speaker_id: dict[str, list[Path]] = defaultdict(list)

    for wav_path in wav_paths:
        stem = wav_path.stem
        by_exact_stem[stem] = wav_path
        by_canonical_stem[canonicalize_part3_stem(stem)].append(wav_path)
        speaker_id = parse_part3_speaker_id_from_stem(stem)
        if speaker_id is not None:
            by_speaker_id[speaker_id].append(wav_path)

    return wav_paths, by_exact_stem, by_canonical_stem, by_speaker_id


def resolve_part3_wav(
    textgrid_stem: str,
    speaker_id: Optional[str],
    by_exact_stem: dict[str, Path],
    by_canonical_stem: dict[str, list[Path]],
    by_speaker_id: dict[str, list[Path]],
) -> Optional[Path]:
    direct = by_exact_stem.get(textgrid_stem)
    if direct is not None:
        return direct

    canonical_matches = by_canonical_stem.get(canonicalize_part3_stem(textgrid_stem), [])
    if len(canonical_matches) == 1:
        return canonical_matches[0]
    if len(canonical_matches) > 1:
        prefers_edited = "_edited" in textgrid_stem.lower()
        preferred_matches = [
            path
            for path in canonical_matches
            if ("_edited" in path.stem.lower()) == prefers_edited
        ]
        if len(preferred_matches) == 1:
            return preferred_matches[0]

    if speaker_id is None:
        return None

    speaker_matches = by_speaker_id.get(speaker_id, [])
    if len(speaker_matches) == 1:
        return speaker_matches[0]

    return None


def parse_textgrid_intervals(textgrid_path: Path):
    intervals = []
    current: Optional[dict[str, object]] = None

    raw = textgrid_path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:200]:
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("intervals ["):
            if (
                current is not None
                and "xmin" in current
                and "xmax" in current
                and "text" in current
            ):
                intervals.append(
                    (float(current["xmin"]), float(current["xmax"]), str(current["text"]))
                )
            current = {}
            continue

        if current is None:
            continue

        if stripped.startswith("xmin ="):
            try:
                current["xmin"] = float(stripped.split("=", 1)[1].strip())
            except ValueError:
                pass
            continue

        if stripped.startswith("xmax ="):
            try:
                current["xmax"] = float(stripped.split("=", 1)[1].strip())
            except ValueError:
                pass
            continue

        if stripped.startswith("text ="):
            text_match = re.match(r'text\s*=\s*"(.*)"\s*$', stripped)
            if text_match:
                current["text"] = text_match.group(1).replace('""', '"')

    if (
        current is not None
        and "xmin" in current
        and "xmax" in current
        and "text" in current
    ):
        intervals.append((float(current["xmin"]), float(current["xmax"]), str(current["text"])))

    return intervals


def collect_part3_from_textgrid(
    split_dir: Path,
    textgrid_dir: Path,
    audio_dir: Path,
    slice_out_dir: Path,
    min_duration_sec: float,
    max_duration_sec: float,
    silence_tokens: set[str],
    overwrite_slices: bool,
):
    split_map = build_split_map(split_dir, 3)
    print(
        "Part 3 (TextGrid mode): split sizes -> "
        f"train={sum(1 for v in split_map.values() if v == 'train')}, "
        f"val={sum(1 for v in split_map.values() if v == 'val')}, "
        f"test={sum(1 for v in split_map.values() if v == 'test')}"
    )

    if not textgrid_dir.exists():
        raise FileNotFoundError(f"Part 3 TextGrid dir not found: {textgrid_dir}")
    textgrid_paths = sorted(path for path in textgrid_dir.iterdir() if path.suffix == ".TextGrid")
    print(f"Part 3 (TextGrid mode): textgrids discovered={len(textgrid_paths)}")

    wav_paths, by_exact_stem, by_canonical_stem, by_speaker_id = index_part3_wavs(audio_dir)
    print(f"Part 3 (TextGrid mode): source WAVs discovered={len(wav_paths)}")

    rows = []
    matched_textgrids = 0
    skipped_unknown_speaker = 0
    missing_or_ambiguous_audio = 0
    unreadable_source_wavs = 0
    malformed_intervals = 0
    filtered_short = 0
    filtered_long = 0
    written_slices = 0
    failed_slice_writes = 0

    silence_tokens_upper = {token.strip().upper() for token in silence_tokens}

    for textgrid_path in tqdm(textgrid_paths, desc="Slicing part3 TextGrids"):
        speaker_id = parse_part3_speaker_id_from_stem(textgrid_path.stem)
        if speaker_id is None:
            skipped_unknown_speaker += 1
            continue

        split = split_map.get(speaker_id)
        if split is None:
            skipped_unknown_speaker += 1
            continue

        source_wav = resolve_part3_wav(
            textgrid_path.stem,
            speaker_id,
            by_exact_stem,
            by_canonical_stem,
            by_speaker_id,
        )
        if source_wav is None:
            missing_or_ambiguous_audio += 1
            continue

        intervals = parse_textgrid_intervals(textgrid_path)
        if not intervals:
            continue

        try:
            audio, sample_rate = sf.read(str(source_wav))
        except Exception:
            unreadable_source_wavs += 1
            continue

        if sample_rate <= 0:
            unreadable_source_wavs += 1
            continue

        num_samples = len(audio)
        matched_textgrids += 1

        for interval_idx, (xmin, xmax, transcript) in enumerate(intervals, start=1):
            transcript_stripped = transcript.strip()
            if not transcript_stripped:
                continue
            if transcript_stripped.upper() in silence_tokens_upper:
                continue

            start_sample = max(0, int(round(xmin * sample_rate)))
            end_sample = min(num_samples, int(round(xmax * sample_rate)))
            if end_sample <= start_sample:
                malformed_intervals += 1
                continue

            duration = (end_sample - start_sample) / sample_rate
            if duration < min_duration_sec:
                filtered_short += 1
                continue
            if duration > max_duration_sec:
                filtered_long += 1
                continue

            utt_id = f"{source_wav.stem}_i{interval_idx:04d}"
            slice_path = slice_out_dir / speaker_id / "wav" / f"{utt_id}.wav"
            slice_path.parent.mkdir(parents=True, exist_ok=True)

            if overwrite_slices or not slice_path.exists():
                segment = audio[start_sample:end_sample]
                try:
                    sf.write(str(slice_path), segment, sample_rate)
                except Exception:
                    failed_slice_writes += 1
                    continue
                written_slices += 1

            rows.append(
                {
                    "part": 3,
                    "speaker_id": speaker_id,
                    "utt_id": utt_id,
                    "path": str(slice_path.resolve()),
                    "duration_sec": round(duration, 3),
                    "split": split,
                    "start_sec": round(start_sample / sample_rate, 3),
                    "end_sec": round(end_sample / sample_rate, 3),
                    "source_path": str(source_wav.resolve()),
                    "textgrid_path": str(textgrid_path.resolve()),
                    "transcript": transcript_stripped,
                }
            )

    print(f"Part 3 (TextGrid mode): matched textgrids={matched_textgrids}")
    print(f"Part 3 (TextGrid mode): skipped unknown speakers={skipped_unknown_speaker}")
    print(f"Part 3 (TextGrid mode): missing/ambiguous audio={missing_or_ambiguous_audio}")
    print(f"Part 3 (TextGrid mode): unreadable source WAVs={unreadable_source_wavs}")
    print(f"Part 3 (TextGrid mode): malformed intervals={malformed_intervals}")
    print(f"Part 3 (TextGrid mode): filtered short (< {min_duration_sec}s)={filtered_short}")
    print(f"Part 3 (TextGrid mode): filtered long  (> {max_duration_sec}s)={filtered_long}")
    print(f"Part 3 (TextGrid mode): slices written={written_slices}")
    print(f"Part 3 (TextGrid mode): failed slice writes={failed_slice_writes}")

    return rows


def build_rows(
    wav_metadata: list[tuple[int, Path, str, str]],
    min_duration_sec: float,
    max_duration_sec: float,
    max_workers: int,
):
    rows = []
    filtered_short = 0
    filtered_long = 0
    unreadable = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(wav_duration_sec, wav): (part, wav, speaker_id, split)
            for part, wav, speaker_id, split in wav_metadata
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing WAVs"):
            part, wav, speaker_id, split = futures[future]
            duration = future.result()
            if duration is None:
                unreadable += 1
                continue
            if duration < min_duration_sec:
                filtered_short += 1
                continue
            if duration > max_duration_sec:
                filtered_long += 1
                continue

            rows.append(
                {
                    "part": part,
                    "speaker_id": speaker_id,
                    "utt_id": wav.stem,
                    "path": str(wav.resolve()),
                    "duration_sec": round(duration, 3),
                    "split": split,
                }
            )

    return rows, filtered_short, filtered_long, unreadable


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build utterance table CSVs from NSC speaker audio directories."
    )
    parser.add_argument(
        "--parts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Which parts to process (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--audio_root_base",
        type=Path,
        default=repo_root,
        help="Base path containing part_{N}/speakers or part{N}/speakers.",
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=repo_root / "data" / "splits",
        help="Directory containing part{N}_train.txt, val, test split files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=repo_root / "data" / "metadata" / "utterances",
        help="Output directory for part{N}_utterances.csv files.",
    )
    parser.add_argument(
        "--speaker_metadata_dir",
        type=Path,
        default=repo_root / "data" / "metadata" / "cleaned",
        help="Directory containing part{N}_speakers.csv for metadata attachment.",
    )
    parser.add_argument(
        "--no_attach_speaker_metadata",
        action="store_true",
        help="Skip merging age/gender/ethnicity into utterance tables.",
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=3.0,
        help="Minimum utterance duration (seconds).",
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=15.0,
        help="Maximum utterance duration (seconds).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=min(8, os.cpu_count() or 8),
        help="Number of worker threads for duration extraction.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write all_utterances.csv by concatenating selected parts.",
    )
    parser.add_argument(
        "--part3_use_textgrid",
        action="store_true",
        help="For part 3, slice speaker-separate conversation WAVs with TextGrid intervals.",
    )
    parser.add_argument(
        "--part3_textgrid_dir",
        type=Path,
        default=repo_root / "data" / "p3_textgrid" / "Scripts Separate",
        help="Directory containing part 3 TextGrid files.",
    )
    parser.add_argument(
        "--part3_audio_dir",
        type=Path,
        default=None,
        help="Directory containing part 3 speaker-separate WAVs (conf_*.wav).",
    )
    parser.add_argument(
        "--part3_slice_out_dir",
        type=Path,
        default=repo_root / "data" / "p3_textgrid" / "sliced_utterances",
        help="Output directory for sliced part 3 utterance WAVs.",
    )
    parser.add_argument(
        "--part3_silence_tokens",
        nargs="+",
        default=["<Z>", "<S>"],
        help="TextGrid interval texts treated as silence for part 3 slicing.",
    )
    parser.add_argument(
        "--part3_overwrite_slices",
        action="store_true",
        help="Overwrite existing part 3 sliced WAVs if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_parts_df = []
    for part in args.parts:
        print(f"\nProcessing part {part}...")
        rows = []

        if part == 3 and args.part3_use_textgrid:
            if args.part3_audio_dir is None:
                print("Part 3: ERROR -> --part3_audio_dir is required with --part3_use_textgrid")
                continue
            try:
                rows = collect_part3_from_textgrid(
                    split_dir=args.split_dir,
                    textgrid_dir=args.part3_textgrid_dir,
                    audio_dir=args.part3_audio_dir,
                    slice_out_dir=args.part3_slice_out_dir,
                    min_duration_sec=args.min_duration_sec,
                    max_duration_sec=args.max_duration_sec,
                    silence_tokens=set(args.part3_silence_tokens),
                    overwrite_slices=args.part3_overwrite_slices,
                )
            except Exception as exc:
                print(f"Part {part}: ERROR -> {exc}")
                continue
        else:
            try:
                wav_metadata = collect_part_wavs(part, args.audio_root_base, args.split_dir)
            except Exception as exc:
                print(f"Part {part}: ERROR -> {exc}")
                continue

            rows, short_count, long_count, unreadable_count = build_rows(
                wav_metadata,
                min_duration_sec=args.min_duration_sec,
                max_duration_sec=args.max_duration_sec,
                max_workers=args.max_workers,
            )

            print(f"Part {part}: unreadable WAVs={unreadable_count}")
            print(f"Part {part}: filtered short (< {args.min_duration_sec}s)={short_count}")
            print(f"Part {part}: filtered long  (> {args.max_duration_sec}s)={long_count}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df["speaker_id"] = df["speaker_id"].astype(str)

            if not args.no_attach_speaker_metadata:
                speaker_meta_path = args.speaker_metadata_dir / f"part{part}_speakers.csv"
                if speaker_meta_path.exists():
                    speaker_meta = pd.read_csv(speaker_meta_path, dtype={"speaker_id": str})
                    if "speaker_id" in speaker_meta.columns:
                        merge_cols = [
                            col
                            for col in ["age", "gender", "ethnicity"]
                            if col in speaker_meta.columns and col not in df.columns
                        ]
                        if merge_cols:
                            speaker_meta = speaker_meta[["speaker_id"] + merge_cols].drop_duplicates(
                                subset=["speaker_id"]
                            )
                            df = df.merge(speaker_meta, on="speaker_id", how="left")
                            print(
                                f"Part {part}: attached speaker metadata columns={merge_cols} "
                                f"from {speaker_meta_path}"
                            )

            df = df.sort_values(["split", "speaker_id", "utt_id"]).reset_index(drop=True)

        out_path = args.out_dir / f"part{part}_utterances.csv"
        df.to_csv(out_path, index=False)

        print(f"Part {part}: wrote {len(df)} utterances -> {out_path}")
        all_parts_df.append(df)

    if args.combined and all_parts_df:
        combined_df = pd.concat(all_parts_df, ignore_index=True)
        if not combined_df.empty:
            combined_df = combined_df.sort_values(["part", "split", "speaker_id", "utt_id"]).reset_index(
                drop=True
            )

        combined_path = args.out_dir / "all_utterances.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nWrote combined utterances: {len(combined_df)} rows -> {combined_path}")


if __name__ == "__main__":
    main()
