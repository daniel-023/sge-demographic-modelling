#!/usr/bin/env python3

"""
PyTorch Dataset classes for loading embeddings and labels.

EmbeddingDataset: For MLP training (mask-aware mean pooling over frames)
SequenceEmbeddingDataset: For LSTM training (keeps frame sequences)

Tasks:
- Classification: gender, ethnicity, age_bin, age_code
- Regression: age_raw
"""

from collections import OrderedDict, defaultdict
from pathlib import Path
import random
import re

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


CLASSIFICATION_TASKS = {"gender", "ethnicity", "age_bin", "age_code"}
REGRESSION_TASKS = {"age_raw"}
SUPPORTED_TASKS = CLASSIFICATION_TASKS | REGRESSION_TASKS
PART_PATTERN = re.compile(r"(part_?\d+)", re.IGNORECASE)


def canonical_task_name(task):
    return task


def task_type_for(task):
    task = canonical_task_name(task)
    if task in REGRESSION_TASKS:
        return "regression"
    if task in CLASSIFICATION_TASKS:
        return "classification"
    raise ValueError(f"Unsupported task: {task}")


def _normalize_speaker_id(speaker_id):
    s = str(speaker_id).strip()
    if s.isdigit():
        return s.zfill(4)
    # Handle IDs like "S001", "PART1_0007", "speaker-42" by taking trailing digits.
    m = re.search(r"(\d+)$", s)
    if m:
        return m.group(1).zfill(4)
    return s


def _age_bin(age):
    if pd.isna(age):
        return "UNKNOWN"
    if isinstance(age, str):
        age_str = age.strip().upper()
        if age_str in {"1X", "2X", "3X", "4X", "5X", "6X"}:
            return age_str
        if age_str == "7X+":
            return age_str
        return "UNKNOWN"
    try:
        age_int = int(age)
        if age_int < 10:
            return "UNKNOWN"
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
    except (ValueError, TypeError):
        return "UNKNOWN"


def _to_numeric_age(age):
    try:
        return float(age)
    except (TypeError, ValueError):
        return None


def _resolve_target_series(task, metadata):
    task = canonical_task_name(task)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    if task == "age_bin":
        if "age_group" in metadata.columns:
            return metadata["age_group"], "age_group", "classification"
        if "age" in metadata.columns:
            return metadata["age"].apply(_age_bin), "age", "classification"
        raise ValueError("Task 'age_bin' requires either 'age_group' or 'age' column in metadata")

    if task == "age_code":
        if "age" not in metadata.columns:
            raise ValueError("Task 'age_code' requires an 'age' column in metadata")
        return metadata["age"], "age", "classification"

    if task == "age_raw":
        if "age" not in metadata.columns:
            raise ValueError("Task 'age_raw' requires an 'age' column in metadata")
        return metadata["age"].apply(_to_numeric_age), "age", "regression"

    if task not in metadata.columns:
        raise ValueError(f"Task '{task}' column not found in metadata")
    return metadata[task], task, "classification"


def _infer_part_tag(*values):
    for value in values:
        if not value:
            continue
        match = PART_PATTERN.search(str(value))
        if match:
            return match.group(1).lower().replace("_", "")
    return None


def _resolve_split_file(split, split_dir, metadata_csv=None, embedding_dir=None):
    split_dir = Path(split_dir)
    candidates = [split_dir / f"{split}.txt"]

    part_tag = _infer_part_tag(metadata_csv, embedding_dir, split_dir)
    if part_tag:
        candidates.append(split_dir / f"{part_tag}_{split}.txt")

    for split_path in candidates:
        if split_path.exists():
            return split_path

    candidate_list = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Metadata has no 'split' column and no matching split file was found. "
        f"Tried: {candidate_list}"
    )


def _load_split_speakers(split, split_dir, metadata_csv=None, embedding_dir=None):
    split_path = _resolve_split_file(
        split,
        split_dir,
        metadata_csv=metadata_csv,
        embedding_dir=embedding_dir,
    )
    return {
        _normalize_speaker_id(line)
        for line in split_path.read_text().splitlines()
        if line.strip()
    }


def _load_chunk_cpu(chunk_file):
    """Load shard file on CPU with mmap when supported by this torch version."""
    try:
        return torch.load(str(chunk_file), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(chunk_file, map_location="cpu")


class ChunkLengthBatchSampler(Sampler):
    """
    Batch sampler that keeps batches within shard boundaries and groups by length.
    This improves shard locality and reduces padding waste.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.chunk_to_indices = defaultdict(list)
        for sample_idx, (chunk_file, _, _) in enumerate(dataset.samples):
            self.chunk_to_indices[str(chunk_file)].append(sample_idx)

    def __iter__(self):
        chunk_keys = list(self.chunk_to_indices.keys())
        if self.shuffle:
            random.shuffle(chunk_keys)

        for chunk_key in chunk_keys:
            indices = list(self.chunk_to_indices[chunk_key])
            indices.sort(key=lambda i: self.dataset.sample_lengths[i])

            batches = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                yield batch

    def __len__(self):
        total = 0
        for indices in self.chunk_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


class _BaseEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_dir,
        split,
        task="gender",
        metadata_csv=None,
        split_dir=None,
        max_cached_chunks=128,
        use_index_cache=True,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.split = split
        self.task = canonical_task_name(task)
        self.metadata_csv = metadata_csv
        self.split_dir = split_dir
        self.task_type = task_type_for(self.task)
        self.max_cached_chunks = max_cached_chunks
        self.use_index_cache = use_index_cache

        if metadata_csv is None:
            raise ValueError("metadata_csv must be provided")

        metadata = pd.read_csv(metadata_csv, dtype={"speaker_id": str})
        if "speaker_id" not in metadata.columns:
            raise ValueError("metadata_csv must contain a 'speaker_id' column")
        metadata["speaker_id"] = metadata["speaker_id"].apply(_normalize_speaker_id)

        target_series, target_name, resolved_task_type = _resolve_target_series(self.task, metadata)
        if resolved_task_type != self.task_type:
            raise ValueError(
                f"Task type mismatch for task '{self.task}': expected {self.task_type}, got {resolved_task_type}"
            )

        metadata = metadata.copy()
        metadata["__target__"] = target_series
        self.target_column = target_name

        if "split" in metadata.columns:
            split_metadata = metadata[metadata["split"] == split]
        elif split_dir is not None:
            split_speakers = _load_split_speakers(
                split,
                split_dir,
                metadata_csv=metadata_csv,
                embedding_dir=self.embedding_dir,
            )
            split_metadata = metadata[metadata["speaker_id"].isin(split_speakers)]
        else:
            raise ValueError(
                "Metadata has no 'split' column. Provide split_dir with train/val/test speaker-id files."
            )

        split_metadata = split_metadata.dropna(subset=["__target__"])

        if self.task_type == "classification":
            # Build a GLOBAL label map from full metadata so class indices are
            # stable across train/val/test splits.
            label_values = sorted(metadata["__target__"].dropna().unique().tolist(), key=str)
            self.label_map = {value: idx for idx, value in enumerate(label_values)}
            self.num_classes = len(self.label_map)
            speaker_to_target = {
                speaker_id: self.label_map[target]
                for speaker_id, target in zip(split_metadata["speaker_id"], split_metadata["__target__"])
            }
        else:
            self.label_map = None
            self.num_classes = 1
            speaker_to_target = {
                speaker_id: float(target)
                for speaker_id, target in zip(split_metadata["speaker_id"], split_metadata["__target__"])
            }

        self.samples = []
        self.sample_speaker_ids = []
        self.sample_lengths = []
        self._chunk_cache = OrderedDict()
        split_embedding_dir = self.embedding_dir / split
        chunk_files = []
        if split_embedding_dir.exists() and split_embedding_dir.is_dir():
            shard_subdir = getattr(self, "shard_subdir", None)
            shard_subdirs = getattr(self, "shard_subdirs", None)
            if shard_subdirs:
                for subdir_name in shard_subdirs:
                    shard_dir = split_embedding_dir / subdir_name
                    if shard_dir.exists() and shard_dir.is_dir():
                        chunk_files = sorted(shard_dir.glob("*.pt"))
                    if chunk_files:
                        break
            elif shard_subdir:
                shard_dir = split_embedding_dir / shard_subdir
                if shard_dir.exists() and shard_dir.is_dir():
                    chunk_files = sorted(shard_dir.glob("*.pt"))
            if not chunk_files:
                chunk_files = sorted(split_embedding_dir.glob("*.pt"))

        if not chunk_files:
            chunk_files = sorted(self.embedding_dir.glob("*.pt"))

        split_file = None
        if "split" not in metadata.columns and split_dir is not None:
            split_file = _resolve_split_file(
                split,
                split_dir,
                metadata_csv=metadata_csv,
                embedding_dir=self.embedding_dir,
            )
        index_cache_path = self._index_cache_path(split_embedding_dir)
        index_cache = self._load_index_cache(
            index_cache_path=index_cache_path,
            chunk_files=chunk_files,
            metadata_csv=metadata_csv,
            split_file=split_file,
        )

        if index_cache is not None:
            self.samples = index_cache["samples"]
            self.sample_speaker_ids = index_cache["sample_speaker_ids"]
            self.sample_lengths = index_cache["sample_lengths"]
            self.embedding_dim = int(index_cache["embedding_dim"])
        else:
            embedding_dim = None
            for chunk_file in chunk_files:
                chunk = _load_chunk_cpu(chunk_file)
                speaker_ids = [_normalize_speaker_id(sid) for sid in chunk["speaker_ids"]]
                feature_key = getattr(self, "feature_key", None)
                if feature_key is None:
                    raise ValueError("Dataset subclass must define feature_key")
                if feature_key not in chunk:
                    raise KeyError(
                        f"Chunk missing expected key '{feature_key}'. Available keys: {list(chunk.keys())}"
                    )
                feature_container = chunk[feature_key]
                lengths = chunk.get("lengths")
                chunk_file_str = str(chunk_file)
                for idx, speaker_id in enumerate(speaker_ids):
                    target = speaker_to_target.get(speaker_id)
                    if target is not None:
                        self.samples.append((chunk_file_str, idx, target))
                        self.sample_speaker_ids.append(speaker_id)
                        if lengths is not None:
                            sample_len = int(lengths[idx].item())
                        else:
                            sample_feat = feature_container[idx]
                            sample_len = int(sample_feat.size(0)) if sample_feat.ndim >= 2 else 1
                        self.sample_lengths.append(sample_len)
                        if embedding_dim is None:
                            sample_feat = feature_container[idx]
                            if sample_feat.ndim == 1:
                                embedding_dim = int(sample_feat.shape[0])
                            elif sample_feat.ndim >= 2:
                                embedding_dim = int(sample_feat.shape[-1])
                            else:
                                raise ValueError(
                                    f"Unsupported embedding shape: {tuple(sample_feat.shape)} in {chunk_file}"
                                )

            if embedding_dim is None:
                raise ValueError(
                    f"No samples found for split='{split}', task='{task}'. Check embedding_dir and metadata alignment."
                )
            self.embedding_dim = embedding_dim
            self._save_index_cache(
                index_cache_path=index_cache_path,
                chunk_files=chunk_files,
                metadata_csv=metadata_csv,
                split_file=split_file,
            )

        if not self.samples:
            raise ValueError(
                f"No samples found for split='{split}', task='{task}'. Check embedding_dir and metadata alignment."
            )

        if not hasattr(self, "embedding_dim"):
            first_chunk = self._get_chunk(self.samples[0][0])
            feature_key = getattr(self, "feature_key", None)
            if feature_key not in first_chunk:
                raise KeyError(
                    f"Chunk missing expected key '{feature_key}'. Available keys: {list(first_chunk.keys())}"
                )
            feature_container = first_chunk[feature_key]
            first_embedding = feature_container[0]
            if first_embedding.ndim == 1:
                self.embedding_dim = int(first_embedding.shape[0])
            elif first_embedding.ndim >= 2:
                self.embedding_dim = int(first_embedding.shape[-1])
            else:
                raise ValueError(
                    f"Unsupported embedding shape: {tuple(first_embedding.shape)} in {self.samples[0][0]}"
                )

    def _index_cache_path(self, split_embedding_dir):
        cache_root = split_embedding_dir if split_embedding_dir.exists() else self.embedding_dir
        cache_dir = cache_root / ".index_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = self.__class__.__name__.lower()
        return cache_dir / f"{dataset_name}_{self.split}_{self.task}.pt"

    def _source_latest_mtime(self, chunk_files, metadata_csv, split_file):
        mtimes = []
        meta_path = Path(metadata_csv)
        if meta_path.exists():
            mtimes.append(meta_path.stat().st_mtime)
        if split_file is not None and split_file.exists():
            mtimes.append(split_file.stat().st_mtime)
        for chunk_file in chunk_files:
            mtimes.append(Path(chunk_file).stat().st_mtime)
        return max(mtimes) if mtimes else 0.0

    def _load_index_cache(self, index_cache_path, chunk_files, metadata_csv, split_file):
        if not self.use_index_cache or not index_cache_path.exists():
            return None
        try:
            cache = torch.load(index_cache_path, map_location="cpu")
        except Exception:
            return None
        if cache.get("version") != 1:
            return None
        if cache.get("task") != self.task or cache.get("split") != self.split:
            return None
        chunk_paths = [str(p) for p in chunk_files]
        if cache.get("chunk_files") != chunk_paths:
            return None
        source_latest_mtime = self._source_latest_mtime(chunk_files, metadata_csv, split_file)
        cached_source_mtime = float(cache.get("source_latest_mtime", -1.0))
        if source_latest_mtime > cached_source_mtime:
            return None
        return cache

    def _save_index_cache(self, index_cache_path, chunk_files, metadata_csv, split_file):
        if not self.use_index_cache:
            return
        payload = {
            "version": 1,
            "task": self.task,
            "split": self.split,
            "chunk_files": [str(p) for p in chunk_files],
            "source_latest_mtime": self._source_latest_mtime(chunk_files, metadata_csv, split_file),
            "samples": self.samples,
            "sample_speaker_ids": self.sample_speaker_ids,
            "sample_lengths": self.sample_lengths,
            "embedding_dim": int(self.embedding_dim),
        }
        try:
            torch.save(payload, index_cache_path)
        except Exception:
            # Training should continue even if cache write fails.
            pass

    def _get_chunk(self, chunk_file):
        key = str(chunk_file)
        if key not in self._chunk_cache:
            self._chunk_cache[key] = _load_chunk_cpu(chunk_file)
            if self.max_cached_chunks is not None and len(self._chunk_cache) > self.max_cached_chunks:
                self._chunk_cache.popitem(last=False)
        else:
            self._chunk_cache.move_to_end(key)
        return self._chunk_cache[key]

    def __len__(self):
        return len(self.samples)


class EmbeddingDataset(_BaseEmbeddingDataset):
    """Dataset with mask-aware mean pooling over frames for MLP training."""
    shard_subdir = "mlp_shards"
    feature_key = "embeddings"

    def __getitem__(self, idx):
        chunk_file, idx_in_chunk, target = self.samples[idx]
        chunk = self._get_chunk(chunk_file)

        feature_container = chunk[self.feature_key]
        embedding = feature_container[idx_in_chunk]
        if embedding.ndim == 1:
            # Already utterance-level pooled embedding.
            pooled = embedding
        else:
            if "lengths" in chunk:
                valid_length = int(chunk["lengths"][idx_in_chunk].item())
                pooled = embedding[:valid_length].mean(dim=0)
            else:
                pooled = embedding.mean(dim=0)

        if self.task_type == "regression":
            target = float(target)
        return pooled, target


class SequenceEmbeddingDataset(_BaseEmbeddingDataset):
    """Dataset with frame sequences for LSTM training."""
    shard_subdirs = ("lstm_packed", "lstm_shards")
    shard_subdir = "lstm_shards"
    feature_key = "sequences"

    def __init__(
        self,
        embedding_dir,
        split,
        task="gender",
        metadata_csv=None,
        split_dir=None,
        max_seq_len=None,
        max_cached_chunks=128,
    ):
        super().__init__(
            embedding_dir=embedding_dir,
            split=split,
            task=task,
            metadata_csv=metadata_csv,
            split_dir=split_dir,
            max_cached_chunks=max_cached_chunks,
        )
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        chunk_file, idx_in_chunk, target = self.samples[idx]
        chunk = self._get_chunk(chunk_file)

        feature_container = chunk[self.feature_key]
        frame_embeddings = feature_container[idx_in_chunk]
        if "lengths" in chunk:
            valid_length = int(chunk["lengths"][idx_in_chunk].item())
            # Clamp to actual embeddings length to avoid out-of-bounds indexing
            valid_length = min(valid_length, frame_embeddings.size(0))
            sequence = frame_embeddings[:valid_length]
        else:
            sequence = frame_embeddings
            valid_length = int(sequence.size(0))

        if self.max_seq_len is not None and sequence.size(0) > self.max_seq_len:
            sequence = sequence[: self.max_seq_len]
            valid_length = min(valid_length, self.max_seq_len)

        if self.task_type == "regression":
            target = float(target)
        return sequence, target, valid_length


def pad_sequence_collate(batch):
    """Pad variable-length sequence batches and keep original lengths."""
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    label_dtype = torch.float if any(isinstance(x, float) for x in labels) else torch.long
    return padded, torch.tensor(labels, dtype=label_dtype), torch.tensor(lengths, dtype=torch.long)
