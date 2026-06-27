#!/usr/bin/env python3

"""
Create stratified train/val/test splits for NSC corpus parts.

Features:
- Part-agnostic: handles all 5 parts independently
- Stratified splitting by gender, ethnicity, and configurable age groups
- Supports both numeric and categorical age formats
- Generates split files for each part
- Optionally generates combined split file
- Uses cleaned metadata from data/metadata/cleaned/
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def _to_age_decade(age):
    """
    Normalize age into decade bins: 1X..6X, 7X+.
    Handles numeric ages and categorical formats like 2X/7X+.
    """
    if pd.isna(age):
        return None

    if isinstance(age, str):
        age_str = age.strip().upper()
        if age_str in {"1X", "2X", "3X", "4X", "5X", "6X"}:
            return age_str
        if age_str in {"7X", "7X+"}:
            return "7X+"
        if age_str.isdigit():
            age = int(age_str)
        else:
            return None

    try:
        age_int = int(age)
    except (ValueError, TypeError):
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


def age_strat_bin(age, scheme="coarse3"):
    """
    Map age to stratification labels.

    scheme='coarse3': young (1X-2X), mid (3X-4X), older (5X-7X+)
    scheme='decade': 1X..7X+
    """
    decade = _to_age_decade(age)
    if decade is None:
        return "UNKNOWN"

    if scheme == "decade":
        return decade

    if decade in {"1X", "2X"}:
        return "young"
    if decade in {"3X", "4X"}:
        return "mid"
    return "older"


def make_strat_col(df):
    """Create stratification column from gender, ethnicity, and age_strat."""
    return df["gender"] + "_" + df["ethnicity"] + "_" + df["age_strat"]


def split_data(df, part_num, random_state=42, test_size=0.3, val_test_split=0.5):
    """
    Perform stratified train/val/test split with robust handling of small strata.
    
    Args:
        df: DataFrame with speaker data
        part_num: Part number (for error messages)
        random_state: Random seed for reproducibility
        test_size: Fraction for test set (0.3 means 70% train, 30% temp)
        val_test_split: Of the temp set, fraction for test (0.5 means 50/50 val/test)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(random_state)
    
    counts = df["strat_col"].value_counts()
    
    # Remove strata with only 1 speaker (can't be stratified)
    strata_to_keep = counts[counts >= 2].index
    removed_count = len(df) - len(df[df["strat_col"].isin(strata_to_keep)])
    
    if removed_count > 0:
        print(f"  Removing {removed_count} speakers from strata with <2 members")
        df = df[df["strat_col"].isin(strata_to_keep)].copy()
        counts = df["strat_col"].value_counts()
    
    # For small datasets, use simple random split instead of stratified
    # This avoids issues with sklearn's stratified split on small strata
    if len(df) < 100 or counts.min() == 2:
        print(f"  Using simple random split (dataset too small for robust stratification)")
        train_mask = np.random.rand(len(df)) < (1 - test_size)
        train_df = df[train_mask].copy()
        temp_df = df[~train_mask].copy()
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["strat_col"],
        )
    
    # For temp split, use simple random if needed
    if len(temp_df) < 20 or (len(temp_df) > 0 and temp_df["strat_col"].value_counts().min() == 1):
        val_mask = np.random.rand(len(temp_df)) < (1 - val_test_split)
        val_df = temp_df[val_mask].copy()
        test_df = temp_df[~val_mask].copy()
    else:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_test_split,
            random_state=random_state,
            stratify=temp_df["strat_col"],
        )
    
    return train_df, val_df, test_df


def split_part(part_num, metadata_dir, output_dir, age_strat_scheme):
    """
    Create splits for a specific part.
    
    Args:
        part_num: Part number (1-5)
        metadata_dir: Directory containing cleaned metadata CSVs
        output_dir: Directory to save split files
        age_strat_scheme: Age scheme for stratification (coarse3 or decade)
    
    Returns:
        Dictionary with split statistics
    """
    input_csv = metadata_dir / f"part{part_num}_speakers.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_csv}")
    
    print(f"\nProcessing Part {part_num}...")
    print(f"  Reading: {input_csv}")
    
    df = pd.read_csv(input_csv, dtype={"speaker_id": str})
    print(f"  Total speakers: {len(df)}")
    
    # Create age stratification labels
    df["age_strat"] = df["age"].apply(lambda age: age_strat_bin(age, age_strat_scheme))
    
    # Create stratification column
    df["strat_col"] = make_strat_col(df)
    
    print(f"  Before filtering: {len(df)} speakers")
    print(f"  Age stratification scheme: {age_strat_scheme}")
    
    # Perform split
    train_df, val_df, test_df = split_data(df, part_num)
    
    print(f"  After filtering: {len(train_df) + len(val_df) + len(test_df)} speakers")
    
    # Extract speaker IDs
    train_set = set(train_df["speaker_id"])
    val_set = set(val_df["speaker_id"])
    test_set = set(test_df["speaker_id"])
    
    # Validate disjoint sets
    assert train_set.isdisjoint(val_set), "Train and val sets overlap!"
    assert train_set.isdisjoint(test_set), "Train and test sets overlap!"
    assert val_set.isdisjoint(test_set), "Val and test sets overlap!"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write split files
    train_file = output_dir / f"part{part_num}_train.txt"
    val_file = output_dir / f"part{part_num}_val.txt"
    test_file = output_dir / f"part{part_num}_test.txt"
    
    train_file.write_text("\n".join(sorted(train_set)) + "\n")
    val_file.write_text("\n".join(sorted(val_set)) + "\n")
    test_file.write_text("\n".join(sorted(test_set)) + "\n")
    
    print(f"  Train: {len(train_set)} speakers -> {train_file.name}")
    print(f"  Val:   {len(val_set)} speakers -> {val_file.name}")
    print(f"  Test:  {len(test_set)} speakers -> {test_file.name}")
    
    return {
        "part": part_num,
        "total": len(df),
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for NSC corpus parts"
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
        '--output_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/splits",
        help='Directory to save split files'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Also create combined split files for all parts'
    )
    parser.add_argument(
        '--age_strat_scheme',
        type=str,
        default='coarse3',
        choices=['coarse3', 'decade'],
        help="Age stratification labels: coarse3=(young/mid/older), decade=(1X..7X+)"
    )
    
    args = parser.parse_args()
    
    stats = []
    all_train = set()
    all_val = set()
    all_test = set()
    
    # Process each part
    for part_num in args.parts:
        try:
            part_stats = split_part(
                part_num,
                args.metadata_dir,
                args.output_dir,
                args.age_strat_scheme,
            )
            stats.append(part_stats)
            
            # Accumulate for combined split
            if args.combined:
                train_file = args.output_dir / f"part{part_num}_train.txt"
                val_file = args.output_dir / f"part{part_num}_val.txt"
                test_file = args.output_dir / f"part{part_num}_test.txt"
                
                all_train.update(train_file.read_text().strip().split('\n'))
                all_val.update(val_file.read_text().strip().split('\n'))
                all_test.update(test_file.read_text().strip().split('\n'))
        
        except Exception as e:
            print(f"  ERROR processing part {part_num}: {e}")
            continue
    
    # Create combined split files
    if args.combined and stats:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "all_train.txt").write_text("\n".join(sorted(all_train)) + "\n")
        (args.output_dir / "all_val.txt").write_text("\n".join(sorted(all_val)) + "\n")
        (args.output_dir / "all_test.txt").write_text("\n".join(sorted(all_test)) + "\n")
        
        print(f"\nCombined splits:")
        print(f"  Train: {len(all_train)} speakers -> all_train.txt")
        print(f"  Val:   {len(all_val)} speakers -> all_val.txt")
        print(f"  Test:  {len(all_test)} speakers -> all_test.txt")
    
    # Print summary
    if stats:
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        total_speakers = sum(s['total'] for s in stats)
        total_train = sum(s['train'] for s in stats)
        total_val = sum(s['val'] for s in stats)
        total_test = sum(s['test'] for s in stats)
        
        for s in stats:
            print(f"Part {s['part']}: {s['train']:4d} train, {s['val']:4d} val, {s['test']:4d} test (total: {s['total']})")
        
        print(f"{'─'*60}")
        print(f"Total: {total_train:4d} train, {total_val:4d} val, {total_test:4d} test (total: {total_speakers})")
        print(f"\n✓ Data splitting complete!")


if __name__ == "__main__":
    main() 
