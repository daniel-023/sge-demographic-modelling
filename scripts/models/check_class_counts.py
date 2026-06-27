#!/usr/bin/env python3

"""Print utterance-level class counts using the same dataset pipeline as training."""

import argparse
from collections import Counter
from pathlib import Path

from scripts.utils.data_loaders import EmbeddingDataset, SequenceEmbeddingDataset, task_type_for


def build_dataset(model_type, embedding_dir, split, task, metadata_csv, split_dir, max_seq_len):
    if model_type == "mlp":
        return EmbeddingDataset(
            embedding_dir=embedding_dir,
            split=split,
            task=task,
            metadata_csv=metadata_csv,
            split_dir=split_dir,
        )

    return SequenceEmbeddingDataset(
        embedding_dir=embedding_dir,
        split=split,
        task=task,
        metadata_csv=metadata_csv,
        split_dir=split_dir,
        max_seq_len=max_seq_len,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["mlp", "lstm"], default="mlp")
    parser.add_argument("--task", choices=["gender", "ethnicity", "age_bin", "age_code", "age_raw"], required=True)
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max_seq_len", type=int, default=None)
    args = parser.parse_args()

    dataset = build_dataset(
        model_type=args.model_type,
        embedding_dir=args.embedding_dir,
        split=args.split,
        task=args.task,
        metadata_csv=args.metadata_csv,
        split_dir=args.split_dir,
        max_seq_len=args.max_seq_len,
    )

    print(f"Model type: {args.model_type}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Total utterances: {len(dataset.samples)}")

    if task_type_for(args.task) == "classification":
        counts = Counter(label for _, _, label in dataset.samples)
        total = sum(counts.values())
        idx_to_label = {idx: label for label, idx in dataset.label_map.items()}

        print("\nUtterance-level class counts:")
        for class_idx in sorted(idx_to_label):
            label_name = idx_to_label[class_idx]
            count = counts.get(class_idx, 0)
            ratio = count / total if total else 0.0
            print(f"{class_idx}\t{label_name}\tcount={count}\tratio={ratio:.4f}")
    else:
        values = [float(target) for _, _, target in dataset.samples]
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        print("\nUtterance-level age_raw summary:")
        print(f"min={min_val:.4f}")
        print(f"max={max_val:.4f}")
        print(f"mean={mean_val:.4f}")


if __name__ == "__main__":
    main()
