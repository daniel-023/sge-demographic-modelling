#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import pandas as pd

def load_split(split_dir: Path, part_num: int, name: str) -> set[str]:
    return set((split_dir / f"part{part_num}_{name}.txt").read_text().split())

def load_preprocessing_splitter():
    splitter_path = Path(__file__).parent.parent / "preprocessing" / "data_split.py"
    spec = importlib.util.spec_from_file_location("preprocessing_splitter", splitter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load splitter module from {splitter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def counts(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    by_split = df.groupby(["split"] + cols).size().reset_index(name="n_speakers")
    all_splits = df.groupby(cols).size().reset_index(name="n_speakers")
    all_splits["split"] = "all"

    out = pd.concat([by_split, all_splits], ignore_index=True)
    out["attribute"] = label
    # make a single "category" column for nicer tables
    out["category"] = out[cols].astype(str).agg("_".join, axis=1)
    return out[["attribute", "split", "category", "n_speakers"]]

def process_part(
    part_num: int,
    metadata_dir: Path,
    split_dir: Path,
    out_dir: Path,
    utterance_dir: Path,
    splitter,
    regenerate_splits: bool,
    age_strat_scheme: str,
):
    speakers_csv = metadata_dir / f"part{part_num}_speakers.csv"
    
    if not speakers_csv.exists():
        print(f"[SKIP] Part {part_num}: {speakers_csv} not found")
        return
    
    print(f"Processing Part {part_num}...")
    df = pd.read_csv(speakers_csv, dtype={"speaker_id": str})
    df["speaker_id"] = df["speaker_id"].str.zfill(4)

    utterances_csv = utterance_dir / f"part{part_num}_utterances.csv"
    if not utterances_csv.exists():
        print(f"[SKIP] Part {part_num}: {utterances_csv} not found")
        return

    utterances_df = pd.read_csv(utterances_csv, dtype={"speaker_id": str})
    available_speaker_ids = set(utterances_df["speaker_id"].astype(str))
    df = df[df["speaker_id"].isin(available_speaker_ids)].copy()

    # Optional: regenerate part split files from preprocessing splitter first.
    if regenerate_splits:
        splitter.split_part(part_num, metadata_dir, split_dir, age_strat_scheme)

    # Use the same age mapping as preprocessing splitter for consistency.
    if "age_group" in df.columns:
        df["age_bin"] = df["age_group"]
    else:
        df["age_bin"] = df["age"].apply(splitter._to_age_decade)
        df["age_bin"] = df["age_bin"].fillna("UNKNOWN")

    train = load_split(split_dir, part_num, "train")
    val = load_split(split_dir, part_num, "val")
    test = load_split(split_dir, part_num, "test")

    def assign_split(sid: str) -> str:
        if sid in train: return "train"
        if sid in val: return "val"
        if sid in test: return "test"
        return "other"

    df["split"] = df["speaker_id"].apply(assign_split)
    df = df[df["split"] != "other"]  # keep only speakers in splits

    rows = []
    rows.append(counts(df, ["gender"], "gender"))
    rows.append(counts(df, ["ethnicity"], "ethnicity"))
    rows.append(counts(df, ["age_bin"], "age_bin"))
    # optional: joint strata table (can be large)
    rows.append(counts(df, ["gender", "ethnicity", "age_bin"], "gender_ethnicity_age"))
    rows.append(
        pd.DataFrame(
            [
                {
                    "attribute": "total_speakers",
                    "split": "all",
                    "category": "ALL",
                    "n_speakers": int(df["speaker_id"].nunique()),
                }
            ]
        )
    )

    out = pd.concat(rows, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"part{part_num}_demographic_distribution.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Wrote → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate demographic distribution statistics for corpus parts"
    )
    parser.add_argument(
        '--parts',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5],
        help='Which parts to process (default: all)'
    )
    parser.add_argument(
        '--metadata_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/metadata/cleaned",
        help='Directory containing cleaned metadata CSVs'
    )
    parser.add_argument(
        '--split_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/splits",
        help='Directory containing split files'
    )
    parser.add_argument(
        '--regenerate_splits',
        action='store_true',
        help='Regenerate split files using preprocessing splitter before stats'
    )
    parser.add_argument(
        '--age_strat_scheme',
        type=str,
        default='decade',
        choices=['coarse3', 'decade'],
        help='Age stratification scheme passed to preprocessing splitter when regenerating splits'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/stats" / "demographic",
        help='Directory to save demographic distribution files'
    )
    parser.add_argument(
        '--utterance_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/metadata/utterances",
        help='Directory containing part{N}_utterances.csv files'
    )
    
    args = parser.parse_args()
    
    splitter = load_preprocessing_splitter()

    for part_num in args.parts:
        try:
            process_part(
                part_num,
                args.metadata_dir,
                args.split_dir,
                args.output_dir,
                args.utterance_dir,
                splitter,
                args.regenerate_splits,
                args.age_strat_scheme,
            )
        except Exception as e:
            print(f"[ERROR] Part {part_num}: {e}")
