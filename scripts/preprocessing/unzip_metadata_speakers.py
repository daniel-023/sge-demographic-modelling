#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import zipfile


def load_metadata_speaker_ids(metadata_csv: Path) -> set[str]:
    with metadata_csv.open(newline="") as handle:
        return {row["speaker_id"].strip().zfill(4) for row in csv.DictReader(handle)}


def find_zip_paths(audio_dir: Path) -> dict[str, Path]:
    zip_paths: dict[str, Path] = {}
    for path in sorted(audio_dir.glob("SPEAKER*.zip")):
        speaker_id = path.stem.removeprefix("SPEAKER")
        zip_paths[speaker_id] = path
    return zip_paths


def extract_zip(zip_path: Path, output_dir: Path, overwrite: bool) -> tuple[str, str]:
    speaker_dir = output_dir / zip_path.stem
    if speaker_dir.exists():
        if not overwrite:
            return zip_path.name, "skipped-existing"
        shutil.rmtree(speaker_dir)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    return zip_path.name, "extracted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only speaker zip files that have metadata rows."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.audio_dir

    metadata_ids = load_metadata_speaker_ids(args.metadata_csv)
    zip_paths = find_zip_paths(args.audio_dir)
    matched_ids = sorted(metadata_ids & set(zip_paths))
    missing_zips = sorted(metadata_ids - set(zip_paths))
    extra_zips = sorted(set(zip_paths) - metadata_ids)

    if args.limit is not None:
        matched_ids = matched_ids[: args.limit]

    print(f"metadata speakers: {len(metadata_ids)}")
    print(f"zip files found: {len(zip_paths)}")
    print(f"matched speakers to extract: {len(matched_ids)}")
    print(f"metadata speakers without zip: {len(missing_zips)}")
    print(f"zip files without metadata: {len(extra_zips)}")
    if missing_zips:
        print("missing zip sample:", ", ".join(missing_zips[:10]))
    if extra_zips:
        print("extra zip sample:", ", ".join(extra_zips[:10]))

    extracted = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(extract_zip, zip_paths[speaker_id], output_dir, args.overwrite): speaker_id
            for speaker_id in matched_ids
        }
        for future in as_completed(futures):
            speaker_id = futures[future]
            try:
                _, status = future.result()
                if status == "extracted":
                    extracted += 1
                else:
                    skipped += 1
            except Exception as exc:  # pragma: no cover
                failures.append((speaker_id, str(exc)))

    print(f"done: extracted={extracted} skipped_existing={skipped} failures={len(failures)}")
    if failures:
        for speaker_id, error in failures[:20]:
            print(f"failure {speaker_id}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
