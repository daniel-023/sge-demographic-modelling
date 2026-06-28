# Pipeline Guide

End-to-end walkthrough for reproducing the experiments. All commands are run from the repository root. Heavy stages such as WavLM feature extraction and LSTM training should be run on a GPU machine or cluster, but the commands are standard Python entrypoints and are not tied to a specific scheduler.

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Stage 1 — Clean Metadata

Normalise speaker metadata from the raw NSC XLSX files. Run once per corpus part.

```bash
python3 -m scripts.preprocessing.clean_metadata --parts 1 2 3
```

**Output:** `data/metadata/cleaned/part{N}_speakers.csv`

Each row is one speaker with columns: `speaker_id`, `gender`, `ethnicity`, `age`.

---

## Stage 2 — Create Train/Val/Test Splits

Generate stratified, speaker-level splits (70/15/15 by default, stratified by gender × ethnicity × age group).

```bash
python3 -m scripts.preprocessing.data_split --parts 1 2 3
```

**Output:** `data/splits/part{N}_{train,val,test}.txt` — one speaker ID per line.

---

## Stage 3 — Build Utterance Table

Index all utterances for a part. Part 3 requires TextGrid slicing (see below).

```bash
# Parts 1 and 2 (speaker-folder layout)
python3 -m scripts.preprocessing.build_utterance_table --parts 1 2

# Part 3 with TextGrid slicing
python3 -m scripts.preprocessing.build_utterance_table \
    --parts 3 \
    --part3_use_textgrid \
    --part3_audio_dir /path/to/part3/conversation_wavs \
    --part3_textgrid_dir "/path/to/part3/Scripts Separate" \
    --part3_slice_out_dir /path/to/output/sliced_utterances
```

**Output:** `data/metadata/utterances/part{N}_utterances.csv`

Utterance tables are generated artifacts and are ignored by git because they can be large.

See [UTTERANCE_TABLE_GUIDE.md](UTTERANCE_TABLE_GUIDE.md) for the full argument reference.

---

## Stage 4 — Extract WavLM Embeddings

Extract `microsoft/wavlm-base-plus` frame embeddings for each split. This stage is GPU-intensive.

```bash
python3 -m scripts.features.feature_extraction \
    --part 1 \
    --split train

python3 -m scripts.features.feature_extraction \
    --part 1 \
    --split val

python3 -m scripts.features.feature_extraction \
    --part 1 \
    --split test
```

**Output:** `results/embeddings/baseplus/part{N}/{split}/`
- `mlp_shards/` — utterance-level pooled embeddings (for MLP)
- `lstm_shards/` — frame sequences (for LSTM)

Repeat for each part and split. The `--outputs` flag can restrict to `mlp` or `lstm` if only one model type is needed.

---

## Stage 5 — Train Models

Example training commands:

```bash
# MLP — gender, Part 1
python3 -m scripts.models.train_mlp \
    --task gender \
    --embedding_dir results/embeddings/baseplus/part1 \
    --metadata_csv data/metadata/cleaned/part1_speakers.csv \
    --split_dir data/splits

# LSTM — ethnicity, Part 1
python3 -m scripts.models.train_lstm \
    --task ethnicity \
    --embedding_dir results/embeddings/baseplus/part1 \
    --metadata_csv data/metadata/cleaned/part1_speakers.csv \
    --split_dir data/splits
```

Supported tasks: `gender`, `ethnicity`, `age_bin`, `age_raw`. The README reports age regression results only, but age-bin classification remains supported for reproducibility.

**Output:** checkpoints under `results/mlp/{embedding_dir_name}/{task}/` or `results/lstm/{embedding_dir_name}_1layer/{task}/` when using the default one-layer LSTM.

Checkpoints are generated artifacts and are ignored by git.

---

## Stage 6 — Evaluate

Evaluate a saved checkpoint on the test split.

```bash
python3 -m scripts.models.evaluate \
    --checkpoint results/mlp/part1/gender/best_mlp_gender.pt \
    --embedding_dir results/embeddings/baseplus/part1 \
    --metadata_csv data/metadata/cleaned/part1_speakers.csv \
    --split_dir data/splits \
    --part part1
```

**Output:** `results/evaluation/part{N}/{model}_{task}_test_metrics.json` + confusion matrix PNG.

---

## Analysis (Optional)

### Corpus and demographic statistics

```bash
python3 -m scripts.analysis.corpus_stats --parts 1 2 3
python3 -m scripts.analysis.demographic_stats --parts 1 2 3
```

### Speaker-level t-SNE visualisation

```bash
# Aggregate speaker embeddings
python3 -m scripts.analysis.speaker_embeddings \
    --mlp-shard-dir results/embeddings/baseplus/part1/train/mlp_shards \
    --metadata-csv data/metadata/cleaned/part1_speakers.csv \
    --out-path results/analysis/speaker_embeddings/part1/speaker_embeddings.pt

# Generate t-SNE plots
python3 -m scripts.analysis.speaker_tsne \
    --emb-path results/analysis/speaker_embeddings/part1/speaker_embeddings.pt \
    --out-dir results/analysis/tsne/part1
```
