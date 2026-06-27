#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def build_corpus_stats(df: pd.DataFrame) -> pd.DataFrame:
    overall_dur = df["duration_sec"].agg(
        n_samples="count",
        total_sec="sum",
        avg_sec="mean",
        min_sec="min",
        max_sec="max",
    )

    overall = pd.DataFrame(
        [
            {
                "split": "all",
                "n_samples": int(overall_dur["n_samples"]),
                "n_speakers": df["speaker_id"].nunique(),
                "total_hours": overall_dur["total_sec"] / 3600,
                "avg_sec": overall_dur["avg_sec"],
                "min_sec": overall_dur["min_sec"],
                "max_sec": overall_dur["max_sec"],
            }
        ]
    )

    by_split = (
        df.groupby("split")
        .agg(
            n_samples=("duration_sec", "count"),
            total_sec=("duration_sec", "sum"),
            avg_sec=("duration_sec", "mean"),
            min_sec=("duration_sec", "min"),
            max_sec=("duration_sec", "max"),
            n_speakers=("speaker_id", "nunique"),
        )
        .reset_index()
    )

    by_split["total_hours"] = by_split["total_sec"] / 3600
    by_split = by_split.drop(columns=["total_sec"])

    stats = pd.concat([overall, by_split], ignore_index=True)
    for col in ["total_hours", "avg_sec", "min_sec", "max_sec"]:
        stats[col] = stats[col].round(3)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate corpus statistics from part utterance metadata."
    )
    parser.add_argument(
        "--parts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Which parts to process (default: all)",
    )
    parser.add_argument(
        "--utterance_dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/metadata/utterances",
        help="Directory containing part{N}_utterances.csv files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/stats" / "corpus",
        help="Directory to save part{N}_corpus_stats.csv files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for part_num in args.parts:
        input_path = args.utterance_dir / f"part{part_num}_utterances.csv"
        if not input_path.exists():
            print(f"[SKIP] Part {part_num}: {input_path} not found")
            continue

        print(f"Processing Part {part_num}...")
        df = pd.read_csv(input_path)
        stats = build_corpus_stats(df)

        output_path = args.output_dir / f"part{part_num}_corpus_stats.csv"
        stats.to_csv(output_path, index=False)
        print(f"[OK] Wrote corpus stats → {output_path}")


if __name__ == "__main__":
    main()