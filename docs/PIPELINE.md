# Pipeline Guide

End-to-end walkthrough for reproducing the experiments. All commands are run from the repository root. Heavy stages (feature extraction, training) are intended for an HPC cluster — see `hpc/` for the corresponding PBS job scripts.

## Prerequisites

Install dependencies (PyTorch and torchaudio must be installed separately for your CUDA version):

```bash
pip install -r requirements.txt
pip install torch torchaudio
```

---

## Stage 1 — Clean Metadata

Normalise speaker metadata from the raw NSC XLSX files. Run once per corpus part.

```bash
python -m scripts.preprocessing.clean_metadata --parts 1 2 3
```

**Output:** `data/metadata/cleaned/part{N}_speakers.csv`

Each row is one speaker with columns: `speaker_id`, `gender`, `ethnicity`, `age`.

---

## Stage 2 — Create Train/Val/Test Splits

Generate stratified, speaker-level splits (70/15/15 by default, stratified by gender × ethnicity × age group).

```bash
python -m scripts.preprocessing.data_split --parts 1 2 3
```

**Output:** `data/splits/part{N}_{train,val,test}.txt` — one speaker ID per line.

---

## Stage 3 — Build Utterance Table

Index all utterances for a part. Part 3 requires TextGrid slicing (see below).

```bash
# Parts 1 and 2 (speaker-folder layout)
python -m scripts.preprocessing.build_utterance_table --parts 1 2

# Part 3 with TextGrid slicing
python -m scripts.preprocessing.build_utterance_table \
    --parts 3 \
    --part3_use_textgrid \
    --part3_audio_dir /path/to/part3/conversation_wavs \
    --part3_textgrid_dir "/path/to/part3/Scripts Separate" \
    --part3_slice_out_dir /path/to/output/sliced_utterances
```

**Output:** `data/metadata/utterances/part{N}_utterances.csv`

See [UTTERANCE_TABLE_GUIDE.md](UTTERANCE_TABLE_GUIDE.md) for the full argument reference.

---

## Stage 4 — Extract WavLM Embeddings

Extract `microsoft/wavlm-base-plus` frame embeddings for each split. This stage is GPU-intensive and should be run on HPC (`hpc/feature_extraction.pbs`).

```bash
python -m scripts.features.feature_extraction \
    --part 1 \
    --split train

python -m scripts.features.feature_extraction \
    --part 1 \
    --split val

python -m scripts.features.feature_extraction \
    --part 1 \
    --split test
```

**Output:** `results/embeddings/baseplus/part{N}/{split}/`
- `mlp_shards/` — utterance-level pooled embeddings (for MLP)
- `lstm_shards/` — frame sequences (for LSTM)

Repeat for each part and split. The `--outputs` flag can restrict to `mlp` or `lstm` if only one model type is needed.

---

## Stage 5 — Train Models

Run on HPC (`hpc/train_mlp.pbs`, `hpc/train_lstm.pbs`). Example local commands:

```bash
# MLP — gender, Part 1
python -m scripts.models.train_mlp \
    --task gender \
    --embedding_dir results/embeddings/baseplus/part1_mlp_full \
    --metadata_csv data/metadata/cleaned/part1_speakers.csv \
    --split_dir data/splits

# LSTM — ethnicity, Part 1
python -m scripts.models.train_lstm \
    --task ethnicity \
    --embedding_dir results/embeddings/baseplus/part1_cap150 \
    --metadata_csv data/metadata/cleaned/part1_speakers.csv \
    --split_dir data/splits
```

Supported tasks: `gender`, `ethnicity`, `age_bin`, `age_raw`.

**Output:** `results/{mlp,lstm}/{embedding_dir_name}/{task}/best_{model}_{task}.pt`

Use `smoke_train_mlp.pbs` / `smoke_train_lstm.pbs` for a quick 1-epoch sanity check before submitting full jobs.

---

## Stage 6 — Evaluate

Evaluate a saved checkpoint on the test split. Run on HPC (`hpc/evaluate_model.pbs`) or locally.

```bash
python -m scripts.models.evaluate \
    --checkpoint results/mlp/part1_mlp_full/gender/best_mlp_gender.pt \
    --embedding_dir results/embeddings/baseplus/part1_mlp_full \
    --metadata_csv data/metadata/cleaned/part1_speakers.csv \
    --split_dir data/splits \
    --part part1
```

**Output:** `results/evaluation/part{N}/{model}_{task}_test_metrics.json` + confusion matrix PNG.

---

## Analysis (Optional)

### Corpus and demographic statistics

```bash
python -m scripts.analysis.corpus_stats --parts 1 2 3
python -m scripts.analysis.demographic_stats --parts 1 2 3
```

### Speaker-level t-SNE visualisation

```bash
# Aggregate speaker embeddings
python -m scripts.analysis.speaker_embeddings \
    --mlp-shard-dir results/embeddings/baseplus/part1_mlp_full/train/mlp_shards \
    --metadata-csv data/metadata/cleaned/part1_speakers.csv \
    --out-path results/analysis/speaker_embeddings/part1/speaker_embeddings.pt

# Generate t-SNE plots
python -m scripts.analysis.speaker_tsne \
    --emb-path results/analysis/speaker_embeddings/part1/speaker_embeddings.pt \
    --out-dir results/analysis/tsne/part1
```

---

## Live Inference

Run inference on a single WAV file without pre-extracted embeddings:

```bash
python -m scripts.inference \
    --checkpoint results/mlp/part1_mlp_full/gender/best_mlp_gender.pt \
    --audio /path/to/speaker.wav
```

WavLM is extracted on-the-fly. The checkpoint is self-describing — no additional configuration needed.
