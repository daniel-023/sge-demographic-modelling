#!/usr/bin/env python3

"""Evaluate trained MLP/LSTM models on a dataset split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from scripts.models.architectures import build_model
from scripts.models.training_utils import get_default_device, predict
from scripts.utils.data_loaders import EmbeddingDataset, SequenceEmbeddingDataset, pad_sequence_collate
from scripts.utils.metrics import compute_metrics, plot_confusion_matrix

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "evaluation"




def _normalize_part(part):
    part_str = str(part).strip().lower()
    if part_str.startswith("part_"):
        part_str = "part" + part_str[len("part_") :]
    return part_str


def _norm_speaker_id(speaker_id):
    speaker_str = str(speaker_id).strip()
    return speaker_str.zfill(4) if speaker_str.isdigit() else speaker_str


def _age_to_bin(age):
    if pd.isna(age):
        return None
    if isinstance(age, str):
        age_str = age.strip().upper()
        if age_str in {"1X", "2X", "3X", "4X", "5X", "6X", "7X+"}:
            return age_str
        if age_str == "7X":
            return "7X+"
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


def _resolve_target_subgroup_column(task, subgroup_columns):
    task_to_subgroup_column = {
        "gender": "gender",
        "ethnicity": "ethnicity",
        "age_bin": "age_bin",
    }
    candidate = task_to_subgroup_column.get(task)
    return candidate if candidate in subgroup_columns else None


def _metric_definitions(task_type):
    if task_type == "regression":
        return {
            "mae": "Mean absolute error over all evaluation samples.",
            "subgroup_metrics.*.*.count": "Number of samples in the subgroup.",
            "subgroup_metrics.*.*.mae": "Mean absolute error within the subgroup.",
        }
    return {
        "accuracy": "Overall exact-match accuracy over all evaluation samples.",
        "f1_macro": "Unweighted mean of per-class F1 scores.",
        "f1_weighted": "Support-weighted mean of per-class F1 scores.",
        "precision_macro": "Unweighted mean of per-class precision scores.",
        "recall_macro": "Unweighted mean of per-class recall scores.",
        "per_class_metrics.*.support": "Number of true samples for the class label.",
        "per_class_metrics.*.precision": "Precision for the class label.",
        "per_class_metrics.*.recall": "Recall for the class label (TP/(TP+FN)).",
        "per_class_metrics.*.f1": "F1 score for the class label.",
        "subgroup_metrics.*.*.count": "Number of samples in the subgroup.",
        "subgroup_metrics.*.*.accuracy": "Accuracy computed only within the subgroup.",
        "subgroup_metrics.*.*.f1_macro": "Macro-F1 computed only within the subgroup.",
        "confusion_matrix.matrix": "Raw confusion-matrix counts (rows=true labels, cols=pred labels).",
        "confusion_matrix.matrix_row_normalized": "Row-normalized confusion matrix where each row sums to 1.",
    }


def _compute_subgroup_metrics(
    labels,
    preds,
    speaker_ids,
    metadata_csv,
    task,
    task_type,
    include_target_subgroup=False,
):
    metadata = pd.read_csv(metadata_csv, dtype={"speaker_id": str})
    metadata["speaker_id"] = metadata["speaker_id"].map(_norm_speaker_id)
    pred_df = pd.DataFrame(
        {
            "speaker_id": [_norm_speaker_id(sid) for sid in speaker_ids],
            "label": labels,
            "pred": preds,
        }
    )
    merged = pred_df.merge(metadata, on="speaker_id", how="left")

    if "age_group" in merged.columns:
        merged["age_bin"] = merged["age_group"]
    elif "age" in merged.columns:
        merged["age_bin"] = merged["age"].apply(_age_to_bin)

    subgroup_columns = [col for col in ["gender", "ethnicity", "age_bin"] if col in merged.columns]
    target_subgroup_col = _resolve_target_subgroup_column(task, subgroup_columns)
    excluded_target_col = None
    if target_subgroup_col and not include_target_subgroup:
        subgroup_columns = [col for col in subgroup_columns if col != target_subgroup_col]
        excluded_target_col = target_subgroup_col

    subgroup_metrics = {}

    for col in subgroup_columns:
        col_metrics = {}
        grouped = merged.dropna(subset=[col]).groupby(col)
        for group_name, group_df in grouped:
            y_true = group_df["label"].tolist()
            y_pred = group_df["pred"].tolist()
            if task_type == "regression":
                abs_errors = [abs(float(t) - float(p)) for t, p in zip(y_true, y_pred)]
                col_metrics[str(group_name)] = {
                    "count": len(group_df),
                    "mae": float(sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
                }
            else:
                col_metrics[str(group_name)] = {
                    "count": len(group_df),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                }
        subgroup_metrics[col] = col_metrics

    return subgroup_metrics, excluded_target_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--part", type=str, default=None, help="Dataset part (e.g., part_1 or part1)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=250,
        help="Print progress every N batches during prediction; set to 0 to disable.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include_target_subgroup",
        action="store_true",
        help=(
            "Include subgroup metrics for the target column when available "
            "(e.g., gender subgroup for gender task)."
        ),
    )
    args = parser.parse_args()

    device = get_default_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = checkpoint.get("model_type")
    task = checkpoint.get("task")
    task_type = checkpoint.get("task_type", "classification")
    model_kwargs = checkpoint.get("model_kwargs")

    if not model_type or not task or not model_kwargs:
        raise ValueError(
            "Checkpoint missing required metadata (model_type/task/model_kwargs). Re-train with updated scripts."
        )

    print(f"Evaluating {model_type} model on task '{task}' ({task_type})\n")

    max_seq_len = checkpoint.get("max_seq_len")
    if model_type == "mlp":
        dataset = EmbeddingDataset(
            args.embedding_dir,
            split=args.split,
            task=task,
            metadata_csv=args.metadata_csv,
            split_dir=args.split_dir,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        dataset = SequenceEmbeddingDataset(
            args.embedding_dir,
            split=args.split,
            task=task,
            metadata_csv=args.metadata_csv,
            split_dir=args.split_dir,
            max_seq_len=max_seq_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=pad_sequence_collate,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_model(model_type, model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    epoch = checkpoint.get("epoch")
    if epoch is not None:
        print(f"Loaded checkpoint from epoch {epoch}")

    val_metric_name = checkpoint.get("val_metric_name")
    val_metric = checkpoint.get("val_metric")
    if val_metric_name and val_metric is not None:
        print(f"Validation {val_metric_name}: {val_metric:.4f}\n")

    preds, labels = predict(
        model,
        dataloader,
        device,
        task_type=task_type,
        progress_interval=args.progress_interval,
        progress_prefix=f"{task}_{args.split}",
    )

    output_dir = args.output_dir
    if args.part:
        output_dir = output_dir / _normalize_part(args.part)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{model_type}_{task}_{args.split}_metrics.json"

    if task_type == "regression":
        mae = sum(abs(float(p) - float(y)) for p, y in zip(preds, labels)) / len(labels)
        if len(dataset.sample_speaker_ids) != len(preds):
            raise ValueError(
                "Prediction count does not match dataset speaker-id count; cannot compute subgroup metrics."
            )
        subgroup_metrics, excluded_target_col = _compute_subgroup_metrics(
            labels=labels,
            preds=preds,
            speaker_ids=dataset.sample_speaker_ids,
            metadata_csv=args.metadata_csv,
            task=task,
            task_type=task_type,
            include_target_subgroup=args.include_target_subgroup,
        )
        results = {
            "mae": mae,
            "subgroup_metrics": subgroup_metrics,
            "subgroup_metrics_includes_target": bool(args.include_target_subgroup),
            "subgroup_metrics_excluded_target_column": excluded_target_col,
            "metric_definitions": _metric_definitions(task_type),
        }
        print("Evaluation Results:")
        print(f"  MAE: {mae:.4f}")
    else:
        metrics = compute_metrics(labels, preds, dataset.num_classes)
        if len(dataset.sample_speaker_ids) != len(preds):
            raise ValueError(
                "Prediction count does not match dataset speaker-id count; cannot compute subgroup metrics."
            )
        subgroup_metrics, excluded_target_col = _compute_subgroup_metrics(
            labels=labels,
            preds=preds,
            speaker_ids=dataset.sample_speaker_ids,
            metadata_csv=args.metadata_csv,
            task=task,
            task_type=task_type,
            include_target_subgroup=args.include_target_subgroup,
        )
        results = metrics
        results["subgroup_metrics"] = subgroup_metrics
        results["subgroup_metrics_includes_target"] = bool(args.include_target_subgroup)
        results["subgroup_metrics_excluded_target_column"] = excluded_target_col
        results["metric_definitions"] = _metric_definitions(task_type)
        print("Evaluation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_macro']:.4f}")

        idx_to_label = None
        label_map = checkpoint.get("label_map") or getattr(dataset, "label_map", None)
        if label_map:
            idx_to_label = {idx: label for label, idx in label_map.items()}

        observed_labels = sorted(set(int(x) for x in labels) | set(int(x) for x in preds))
        if idx_to_label:
            configured_labels = sorted(int(idx) for idx in idx_to_label.keys())
        else:
            configured_labels = list(range(int(dataset.num_classes)))
        ordered_labels = sorted(set(configured_labels) | set(observed_labels))

        if idx_to_label:
            class_names = [str(idx_to_label.get(i, i)) for i in ordered_labels]
        else:
            class_names = [str(i) for i in ordered_labels]

        precision, recall, f1_vals, support = precision_recall_fscore_support(
            labels, preds, labels=ordered_labels, zero_division=0
        )
        per_class_metrics = {}
        for i, cls_idx in enumerate(ordered_labels):
            cls_name = class_names[i]
            per_class_metrics[cls_name] = {
                "class_index": int(cls_idx),
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1_vals[i]),
            }
        results["per_class_metrics"] = per_class_metrics

        cm = confusion_matrix(labels, preds, labels=ordered_labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_row_normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        results["confusion_matrix"] = {
            "labels": class_names,
            "class_indices": [int(x) for x in ordered_labels],
            "matrix": cm.tolist(),
            "matrix_row_normalized": cm_row_normalized.tolist(),
        }

        cm_path = output_dir / f"{model_type}_{task}.png"
        plot_confusion_matrix(
            labels,
            preds,
            class_names=class_names,
            labels=ordered_labels,
            output_path=cm_path,
            normalize="true",
            annotate_counts=False,
            annotate_percentages=True,
            percentage_symbol=True,
            annotation_decimals=1,
            show_title=True,
            title_text=model_type.upper(),
            show_colorbar=False,
        )

    with result_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
