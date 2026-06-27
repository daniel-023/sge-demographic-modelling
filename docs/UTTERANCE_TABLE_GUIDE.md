# Utterance Table Script - Usage Guide

The `build_utterance_table.py` script is **part-agnostic** and can process all 5 NSC corpus parts.
It now also supports **Part 3 TextGrid slicing** for speaker-separate conversational audio.

## Requirements

**Directory structure expected:**
```
sge-demographic-modelling/
├── part_1/
│   └── speakers/
│       ├── speaker_0001/
│       │   ├── utt_001.WAV
│       │   └── utt_002.WAV
│       └── speaker_0002/
│           └── utt_003.WAV
├── part_2/
│   └── speakers/
│       └── ...
├── part_3/
│   └── speakers/
│       └── ...
└── ... (parts 4, 5)

data/
├── splits/
│   ├── part1_train.txt
│   ├── part1_val.txt
│   ├── part1_test.txt
│   └── ... (all parts)
└── metadata/
    └── cleaned/
        ├── part1_speakers.csv
        ├── part2_speakers.csv
        └── ... (all parts)
```

## Usage

### Process all parts (with combined utterance table)
```bash
cd scripts/preprocessing
python build_utterance_table.py --combined
```

### Process specific parts
```bash
python build_utterance_table.py --parts 1 2 3
```

### Custom audio directory for speaker-folder parts
```bash
python build_utterance_table.py --audio_root_base /path/to/audio/root
```

### Part 3 TextGrid slicing mode

Use this when Part 3 source audio files are long speaker-separate recordings, e.g.
`conf_2500_2500_00862025.wav`, and you want utterance-level WAVs from TextGrid intervals.

```bash
python build_utterance_table.py \
  --parts 3 \
  --part3_use_textgrid \
  --part3_audio_dir /path/to/part3/conversation_wavs \
  --part3_textgrid_dir "/Users/daniel/Documents/VSCode/NTU-LMS-FYP/sge-demographic-modelling/data/p3_textgrid/Scripts Separate" \
  --part3_slice_out_dir /path/to/output/sliced_utterances
```

Optional flags:
- `--part3_silence_tokens "<Z>" "<S>"`: labels treated as silence
- `--part3_overwrite_slices`: rewrite already-sliced WAVs
- `--min_duration_sec` / `--max_duration_sec`: duration filtering applied during slicing
- `--no_attach_speaker_metadata`: do not merge age/gender/ethnicity from `part3_speakers.csv`

## Output

For each part, generates: `data/metadata/cleaned/part{N}_utterances.csv`

With columns:
- `speaker_id`: speaker ID matching split files
- `utt_id`: Utterance ID (usually WAV filename stem)
- `path`: Full path to WAV file
- `duration_sec`: Duration in seconds
- `split`: Train/val/test assignment
- `age`, `gender`, `ethnicity`: merged from `data/metadata/cleaned/part{N}_speakers.csv` (default)

For Part 3 TextGrid mode, additional columns are included:
- `start_sec`
- `end_sec`
- `source_path`
- `textgrid_path`
- `transcript`

With `--combined`: Also generates `all_utterances.csv` containing all utterances from all parts.
