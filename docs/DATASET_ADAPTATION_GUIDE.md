# Dataset Adaptation Guide

This guide describes how to adapt the pipeline in this repository to a new speech dataset. The workflow is organized around four stages:

1. prepare speaker metadata
2. create speaker-level train/validation/test splits
3. extract WavLM embeddings
4. train and evaluate downstream MLP or LSTM models

## 1. Prepare Metadata

The training scripts expect speaker-level metadata in CSV format. Each row should correspond to one speaker.

For full reuse of the current NSC-style pipeline, the metadata should include:

- `speaker_id`
- `gender`
- `ethnicity`
- `age`

However, a new dataset does not need to include all three demographic targets. You can train only the tasks supported by the labels you actually have. For example:

- if the dataset only includes `gender`, you can train `gender`
- if the dataset includes `gender` and `age`, you can train those tasks and skip `ethnicity`
- if the dataset does not include a label for a task, that task should simply be omitted

The current preprocessing flow standardizes metadata into files such as:

- `data/metadata/cleaned/part1_speakers.csv`
- `data/metadata/cleaned/part2_speakers.csv`

The metadata cleaning script is [`clean_metadata.py`](../scripts/preprocessing/clean_metadata.py). It normalizes:

- speaker IDs
- gender labels
- ethnicity labels
- age values

Notes:

- `speaker_id` should match the IDs used in split files and utterance tables.
- classification tasks expect categorical labels such as `gender` and `ethnicity`.
- age can be numeric for regression (`age_raw`) or categorical for age-group classification.
- not every new dataset will support every task; task availability depends on which labels are present and usable

## 2. Create Train/Validation/Test Splits

Splits are speaker-based, not utterance-based. This is important to avoid speaker leakage across train, validation, and test.

The split generation script is [`data_split.py`](../scripts/preprocessing/data_split.py). It creates files such as:

- `data/splits/part1_train.txt`
- `data/splits/part1_val.txt`
- `data/splits/part1_test.txt`

Each split file contains one `speaker_id` per line.

The script stratifies by:

- gender
- ethnicity
- age-derived strata

For a new dataset, the main requirement is that the metadata CSV contains a consistent `speaker_id` column and whichever target labels you plan to use.

Important:

- the current split script is written around the NSC setup and assumes access to `gender`, `ethnicity`, and `age`
- if one or more of those labels are missing in your new dataset, the stratification logic should be adapted to use only the available columns
- speaker-level disjointness across train, validation, and test should still be preserved even if stratification changes

## 3. Build an Utterance Table

Feature extraction operates on utterance-level metadata. The utterance table script is [`build_utterance_table.py`](../scripts/preprocessing/build_utterance_table.py).

Typical utterance table columns include:

- `speaker_id`
- `utt_id`
- `path`
- `duration_sec`
- `split`

Depending on the pipeline mode, additional metadata columns may also be attached.

If your new dataset requires special handling at the utterance level, see:

- [`UTTERANCE_TABLE_GUIDE.md`](/workspace/sge-demographic-modelling/docs/UTTERANCE_TABLE_GUIDE.md)

## 4. Extract Embeddings

Embedding extraction is handled by [`feature_extraction.py`](/workspace/sge-demographic-modelling/scripts/features/feature_extraction.py).

This script supports:

- MLP-style pooled utterance embeddings
- LSTM-style sequence embeddings

By default, it expects:

- an utterance CSV such as `data/metadata/utterances/part{N}_utterances.csv`
- an output directory such as `results/embeddings/baseplus/part{N}`

Important extraction settings:

- `--outputs both|mlp|lstm`
- `--split train|val|test`
- `--max-frames` for sequence truncation
- `--lstm-max-utts-per-speaker` for LSTM utterance caps

For a new dataset, make sure:

- audio paths in the utterance table are valid
- audio can be loaded by `torchaudio`
- `speaker_id` and `split` remain aligned with your metadata and split files

## 5. Train Downstream Models

### MLP

Use [`train_mlp.py`](/workspace/sge-demographic-modelling/scripts/models/train_mlp.py) for pooled embeddings.

Required inputs:

- `--embedding_dir`
- `--metadata_csv`
- `--split_dir`
- `--task`

Supported tasks:

- `gender`
- `ethnicity`
- `age_bin`
- `age_code`
- `age_raw`

Example:

```bash
python -m scripts.models.train_mlp \
  --task gender \
  --embedding_dir results/embeddings/baseplus/part1 \
  --metadata_csv data/metadata/cleaned/part1_speakers.csv \
  --split_dir data/splits \
  --evaluate
```

### LSTM

Use [`train_lstm.py`](/workspace/sge-demographic-modelling/scripts/models/train_lstm.py) for sequence embeddings.

Required inputs:

- `--embedding_dir`
- `--metadata_csv`
- `--split_dir`
- `--task`

Example:

```bash
python -m scripts.models.train_lstm \
  --task ethnicity \
  --embedding_dir results/embeddings/baseplus/part1 \
  --metadata_csv data/metadata/cleaned/part1_speakers.csv \
  --split_dir data/splits \
  --evaluate
```

The current defaults are:

- MLP output directory: `results/mlp/<embedding_dir_name>/<task>`
- LSTM output directory: `results/lstm/<embedding_dir_name>/<task>`
- for 1-layer LSTM, outputs default to `results/lstm/<embedding_dir_name>_1layer/<task>`

## 6. Evaluate Saved Checkpoints

Standalone evaluation is handled by [`evaluate.py`](/workspace/sge-demographic-modelling/scripts/models/evaluate.py).

This script loads a saved checkpoint and computes test metrics on a specified split. Outputs are saved under:

- `results/evaluation/`

Example:

```bash
python -m scripts.models.evaluate \
  --checkpoint results/lstm/part1_cap150/ethnicity/best_lstm_ethnicity.pt \
  --embedding_dir results/embeddings/baseplus/part1 \
  --metadata_csv data/metadata/cleaned/part1_speakers.csv \
  --split_dir data/splits \
  --part part1
```

## Practical Checklist for a New Dataset

Before running training, confirm that:

- metadata has `speaker_id` and the labels required for the task or tasks you want to train
- split files contain speaker IDs that match the metadata
- utterance table rows point to real audio files
- utterance rows can be mapped back to speaker IDs
- extracted embeddings are written under a consistent `embedding_dir`
- the chosen task label exists and is clean enough for training

## When Adaptation Is Needed

You will likely need to modify the pipeline if your new dataset differs substantially from the NSC-based setup, for example:

- different raw metadata column names
- different age label formats
- no ethnicity labels
- only a subset of demographic labels
- no speaker-level split files yet
- long-form recordings that need segmentation before utterance-level processing

In those cases, the most common files to adapt are:

- [`clean_metadata.py`](../scripts/preprocessing/clean_metadata.py)
- [`data_split.py`](../scripts/preprocessing/data_split.py)
- [`build_utterance_table.py`](../scripts/preprocessing/build_utterance_table.py)
- [`feature_extraction.py`](../scripts/features/feature_extraction.py)
