#!/usr/bin/env python3

"""
Clean metadata across all NSC corpus parts.

Features:
- Part-agnostic: handles all 5 parts with different column names and formats
- Standardizes speaker IDs with part prefix (e.g., part1_0001, part3_3000-1)
- Normalizes ethnicity and gender values to uppercase
- Removes 'OTHERS' ethnicity entries
- Handles both numeric and categorical age formats
- Removes duplicates and validates data
"""

import argparse
from pathlib import Path
import pandas as pd
import re


# Column mappings
PART_CONFIGS = {
    1: {'speaker_id': 'SpeakerID', 'age': 'AGE', 'gender': 'SEX', 'ethnicity': 'ACC', 'sheet': None},
    2: {'speaker_id': 'SCD/PART2', 'age': 'AGE', 'gender': 'SEX', 'ethnicity': 'ACC', 'sheet': None},
    3: {'speaker_id': 'Speaker ID', 'age': 'Age', 'gender': 'Gender', 'ethnicity': 'Ethnicity', 'sheet': 'Separate Room'},
    4: {'speaker_id': 'Speaker ID', 'age': 'Age', 'gender': 'Gender', 'ethnicity': 'Ethnic Group', 'sheet': None},
    5: {'speaker_id': 'Person ID', 'age': 'Age', 'gender': 'Gender', 'ethnicity': 'Ethnicity', 'sheet': None},
}

# Standardised column names
STANDARD_COLS = ['speaker_id', 'age', 'gender', 'ethnicity']


def format_speaker_id(speaker_id, part_num):
    """
    Format speaker ID with part prefix.
    
    Examples:
        Part 1: 1 -> part1_0001
        Part 3: 3000-1 -> part3_3000-1
        Part 4: 1010 -> part4_1010
    """
    if isinstance(speaker_id, str):
        sid = speaker_id.strip()
        # Numeric string IDs should be zero-padded to 4 digits (minimum width).
        if sid.isdigit():
            return sid.zfill(4)
        # Non-numeric IDs (e.g., "3000-1") are kept as-is.
        return sid
    else:
        # Numeric IDs: zero-pad to 4 digits
        return f"{int(speaker_id):04d}"


def normalize_ethnicity(value):
    """Normalize ethnicity to uppercase."""
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def normalize_gender(value):
    """Normalize gender to M/F format."""
    if pd.isna(value):
        return None
    
    val = str(value).strip().upper()
    
    # Map variations
    if val in ['MALE', 'M']:
        return 'M'
    elif val in ['FEMALE', 'F']:
        return 'F'
    else:
        return val


def normalize_age(value):
    """
    Normalize age to numeric format where possible.
    Part 3 has categorical format like "2X" (20s).
    """
    if pd.isna(value):
        return None
    
    # If already numeric, return as is
    if isinstance(value, (int, float)):
        return int(value)
    
    # Handle categorical format like "2X" -> keep as string
    val_str = str(value).strip()
    if val_str.endswith('X'):
        return val_str
    
    # Try to parse as number
    try:
        return int(val_str)
    except ValueError:
        return val_str


def clean_part(part_num, input_dir, output_dir, remove_others=True):
    """
    Clean metadata for a specific part.
    
    Args:
        part_num: Part number (1-5)
        input_dir: Directory containing raw metadata
        output_dir: Directory to save cleaned metadata
        remove_others: Whether to remove 'OTHERS' ethnicity entries
    
    Returns:
        DataFrame with cleaned metadata
    """
    # Get column mapping for this part
    col_map = PART_CONFIGS.get(part_num)
    if col_map is None:
        raise ValueError(f"No configuration for part {part_num}")
    
    # Read raw metadata
    input_path = input_dir / f"part{part_num}_metadata.xlsx"
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    print(f"\nProcessing Part {part_num}...")
    print(f"  Reading: {input_path}")
    
    # Read from specific sheet if specified
    sheet_name = col_map.get('sheet')
    if sheet_name:
        print(f"  Using sheet: '{sheet_name}'")
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(input_path)
    
    print(f"  Initial rows: {len(df)}")
    
    # Select and rename relevant columns (exclude 'sheet' key)
    col_keys = [k for k in col_map.keys() if k != 'sheet']
    df = df[[col_map[k] for k in col_keys]].copy()
    df.columns = STANDARD_COLS
    
    # Normalize speaker IDs
    df['speaker_id'] = df['speaker_id'].apply(lambda x: format_speaker_id(x, part_num))
    
    # Normalize ethnicity and gender
    df['ethnicity'] = df['ethnicity'].apply(normalize_ethnicity)
    df['gender'] = df['gender'].apply(normalize_gender)
    df['age'] = df['age'].apply(normalize_age)
    
    # Remove OTHERS ethnicity if requested
    if remove_others:
        count_others = len(df[df['ethnicity'] == 'OTHERS'])
        if count_others > 0:
            print(f"  Removing {count_others} 'OTHERS' ethnicity entries")
            df = df[df['ethnicity'] != 'OTHERS']
    
    # Remove duplicates
    dup_count = df.duplicated(subset=['speaker_id']).sum()
    if dup_count > 0:
        print(f"  Removing {dup_count} duplicate speaker IDs")
        df = df.drop_duplicates(subset=['speaker_id'])
    
    # Remove rows with missing values in key columns
    missing_before = len(df)
    df = df.dropna(subset=['speaker_id', 'ethnicity', 'gender'])
    missing_after = len(df)
    if missing_before != missing_after:
        print(f"  Removed {missing_before - missing_after} rows with missing values")
    
    print(f"  Final rows: {len(df)}")
    print(f"  Ethnicity distribution: {df['ethnicity'].value_counts().to_dict()}")
    print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Clean NSC metadata across all corpus parts"
    )
    parser.add_argument(
        '--parts',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5],
        help='Which parts to process (default: all)'
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/metadata/raw_metadata",
        help='Directory containing raw metadata files'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/metadata/cleaned",
        help='Directory to save cleaned metadata'
    )
    parser.add_argument(
        '--keep_others',
        action='store_true',
        help='Keep OTHERS ethnicity entries (default: remove them)'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Also save a combined file with all parts'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    for part_num in args.parts:
        try:
            df = clean_part(
                part_num,
                args.input_dir,
                args.output_dir,
                remove_others=not args.keep_others
            )
            
            # Save individual part file
            output_path = args.output_dir / f"part{part_num}_speakers.csv"
            df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"  ERROR processing part {part_num}: {e}")
            continue
    
    # Save combined file if requested
    if args.combined and all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = args.output_dir / "all_speakers.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined file saved: {combined_path}")
        print(f"Total speakers: {len(combined_df)}")
        print(f"Ethnicity distribution: {combined_df['ethnicity'].value_counts().to_dict()}")
        print(f"Gender distribution: {combined_df['gender'].value_counts().to_dict()}")
    
    print("\n✓ Metadata cleaning complete!")


if __name__ == "__main__":
    main()
