#!/usr/bin/env python3
"""
Create speaker-level stratified subset splits from existing train/val/test files.

Default behavior:
- Subset only train speakers (speaker-stratified).
- Keep val/test unchanged.
- Write output split files as train.txt / val.txt / test.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_speaker_id(speaker_id):
    s = str(speaker_id).strip()
    return s.zfill(4) if s.isdigit() else s


def _to_age_decade(age):
    if pd.isna(age):
        return None
    if isinstance(age, str):
        age_str = age.strip().upper()
        if age_str in {"1X", "2X", "3X", "4X", "5X", "6X", "7X", "7X+"}:
            return "7X+" if age_str in {"7X", "7X+"} else age_str
        if age_str.isdigit():
            age = int(age_str)
        else:
            return None
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return None
    if age_int < 10:
        return None
    if age_int < 20:
        return "1X"
    if age_int < 30:
        return "2X"
    if age_int < 40:
        return "3X"
    if age_int < 50:
        return "4X"
    if age_int < 60:
        return "5X"
    if age_int < 70:
        return "6X"
    return "7X+"


def _resolve_split_file(split_dir: Path, split: str, part: int | None):
    direct = split_dir / f"{split}.txt"
    if direct.exists():
        return direct
    if part is not None:
        part_file = split_dir / f"part{part}_{split}.txt"
        if part_file.exists():
            return part_file
    raise FileNotFoundError(
        f"Could not find split file for '{split}' in {split_dir}. "
        f"Checked: {direct}" + (f" and {split_dir / f'part{part}_{split}.txt'}" if part is not None else "")
    )


def _read_split_ids(path: Path):
    return [_normalize_speaker_id(x) for x in path.read_text().splitlines() if x.strip()]


def _write_split_ids(path: Path, speaker_ids):
    unique_sorted = sorted(set(_normalize_speaker_id(x) for x in speaker_ids))
    path.write_text("\n".join(unique_sorted) + "\n")


def _resolve_strat_col(df: pd.DataFrame, stratify_on: str):
    if stratify_on == "age_bin":
        if "age_group" in df.columns:
            return df["age_group"].fillna("UNKNOWN").astype(str)
        if "age" in df.columns:
            return df["age"].apply(_to_age_decade).fillna("UNKNOWN").astype(str)
        raise ValueError("age_bin stratification requires age_group or age column in metadata")
    if stratify_on not in df.columns:
        raise ValueError(f"Column '{stratify_on}' not found in metadata")
    return df[stratify_on].fillna("UNKNOWN").astype(str)


def _stratified_subset(df: pd.DataFrame, strat_col: str, fraction: float, seed: int):
    if fraction >= 1.0:
        return df.copy()
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0")

    target_n = max(1, int(round(len(df) * fraction)))
    if target_n >= len(df):
        return df.copy()

    rng = np.random.default_rng(seed)
    grouped = []
    counts = df[strat_col].value_counts()

    # Proportional allocation with exact-size correction.
    raw_alloc = {k: counts[k] * target_n / len(df) for k in counts.index}
    alloc = {k: int(np.floor(v)) for k, v in raw_alloc.items()}
    alloc = {k: min(alloc[k], int(counts[k])) for k in alloc}

    # Ensure represented strata can contribute at least one sample when possible.
    for k, cnt in counts.items():
        if cnt > 0 and alloc[k] == 0:
            alloc[k] = 1

    current = sum(alloc.values())
    if current > target_n:
        # Remove from largest allocations first.
        for k in sorted(alloc, key=lambda x: alloc[x], reverse=True):
            while current > target_n and alloc[k] > 1:
                alloc[k] -= 1
                current -= 1
            if current == target_n:
                break
    elif current < target_n:
        # Add by largest residual while capacity remains.
        residual_order = sorted(raw_alloc, key=lambda x: raw_alloc[x] - alloc[x], reverse=True)
        for k in residual_order:
            while current < target_n and alloc[k] < int(counts[k]):
                alloc[k] += 1
                current += 1
            if current == target_n:
                break

    for k, n in alloc.items():
        if n <= 0:
            continue
        sub = df[df[strat_col] == k]
        if n >= len(sub):
            grouped.append(sub)
        else:
            picked_idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
            grouped.append(sub.loc[picked_idx])

    out = pd.concat(grouped, ignore_index=False)
    # Guard exact size (rare edge cases after allocation correction).
    if len(out) > target_n:
        out = out.sample(n=target_n, random_state=seed)
    elif len(out) < target_n:
        remainder = df.drop(index=out.index)
        need = min(target_n - len(out), len(remainder))
        if need > 0:
            out = pd.concat([out, remainder.sample(n=need, random_state=seed)], ignore_index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Create speaker-stratified subset splits from existing splits")
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--input_split_dir", type=Path, required=True)
    parser.add_argument("--output_split_dir", type=Path, required=True)
    parser.add_argument("--part", type=int, default=None, help="Part number for part{N}_*.txt naming fallback")
    parser.add_argument("--stratify_on", type=str, default="ethnicity", choices=["gender", "ethnicity", "age_bin"])
    parser.add_argument("--train_fraction", type=float, default=0.2)
    parser.add_argument("--subset_val", action="store_true", help="Also subset val split")
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv, dtype={"speaker_id": str})
    if "speaker_id" not in metadata.columns:
        raise ValueError("metadata_csv must contain speaker_id column")
    metadata["speaker_id"] = metadata["speaker_id"].map(_normalize_speaker_id)
    metadata["__strat__"] = _resolve_strat_col(metadata, args.stratify_on)

    train_path = _resolve_split_file(args.input_split_dir, "train", args.part)
    val_path = _resolve_split_file(args.input_split_dir, "val", args.part)
    test_path = _resolve_split_file(args.input_split_dir, "test", args.part)

    train_ids = _read_split_ids(train_path)
    val_ids = _read_split_ids(val_path)
    test_ids = _read_split_ids(test_path)

    train_df = metadata[metadata["speaker_id"].isin(train_ids)].copy()
    if train_df.empty:
        raise ValueError("No train speakers matched metadata")
    train_sub = _stratified_subset(train_df, "__strat__", args.train_fraction, args.seed)
    out_train_ids = train_sub["speaker_id"].tolist()

    if args.subset_val and args.val_fraction < 1.0:
        val_df = metadata[metadata["speaker_id"].isin(val_ids)].copy()
        if val_df.empty:
            raise ValueError("No val speakers matched metadata")
        val_sub = _stratified_subset(val_df, "__strat__", args.val_fraction, args.seed + 1)
        out_val_ids = val_sub["speaker_id"].tolist()
    else:
        out_val_ids = val_ids

    # Keep test fixed by default to avoid evaluation drift.
    out_test_ids = test_ids

    args.output_split_dir.mkdir(parents=True, exist_ok=True)
    _write_split_ids(args.output_split_dir / "train.txt", out_train_ids)
    _write_split_ids(args.output_split_dir / "val.txt", out_val_ids)
    _write_split_ids(args.output_split_dir / "test.txt", out_test_ids)

    print("[OK] Wrote subset splits:")
    print(f"  train: {len(set(out_train_ids))} speakers (fraction={args.train_fraction:.3f})")
    if args.subset_val and args.val_fraction < 1.0:
        print(f"  val:   {len(set(out_val_ids))} speakers (fraction={args.val_fraction:.3f})")
    else:
        print(f"  val:   {len(set(out_val_ids))} speakers (unchanged)")
    print(f"  test:  {len(set(out_test_ids))} speakers (unchanged)")
    print(f"  stratify_on: {args.stratify_on}")
    print(f"  output_dir:  {args.output_split_dir}")


if __name__ == "__main__":
    main()
