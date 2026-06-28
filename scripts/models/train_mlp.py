#!/usr/bin/env python3

"""Train MLP model for demographic prediction from WavLM embeddings."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.models.architectures.mlp import DemographicMLP
from scripts.models.training_utils import (
    build_training_criterion,
    build_weighted_sampler,
    get_default_device,
    predict,
    train_epoch,
    validate,
)
from scripts.utils.data_loaders import EmbeddingDataset, task_type_for
from scripts.utils.metrics import compute_metrics

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["gender", "ethnicity", "age_bin", "age_raw"],
    )
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, default=None)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--balanced_sampling",
        dest="balanced_sampling",
        action="store_true",
        help="Use weighted sampling for classification training",
    )
    parser.add_argument(
        "--no_balanced_sampling",
        dest="balanced_sampling",
        action="store_false",
        help="Disable weighted sampling",
    )
    parser.add_argument(
        "--class_weighted_loss",
        dest="class_weighted_loss",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss for classification",
    )
    parser.add_argument(
        "--no_class_weighted_loss",
        dest="class_weighted_loss",
        action="store_false",
        help="Disable class-weighted CrossEntropyLoss",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["accuracy", "f1_macro"],
        default="f1_macro",
        help="Validation metric used to select best checkpoint (classification only)",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best model on test split")
    parser.add_argument("--test_split", type=str, default="test", help="Dataset split for evaluation")
    parser.set_defaults(balanced_sampling=True, class_weighted_loss=True)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = DEFAULT_RESULTS_ROOT / "mlp" / args.embedding_dir.name / args.task

    task_type = task_type_for(args.task)
    device = get_default_device()
    print(f"Using device: {device}")

    train_dataset = EmbeddingDataset(
        args.embedding_dir,
        split="train",
        task=args.task,
        metadata_csv=args.metadata_csv,
        split_dir=args.split_dir,
    )
    val_dataset = EmbeddingDataset(
        args.embedding_dir,
        split="val",
        task=args.task,
        metadata_csv=args.metadata_csv,
        split_dir=args.split_dir,
    )

    pin_memory = device.type == "cuda"
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.balanced_sampling and task_type == "classification":
        train_loader_kwargs["sampler"] = build_weighted_sampler(
            train_dataset, num_classes=train_dataset.num_classes
        )
        train_loader_kwargs["shuffle"] = False
    else:
        if args.balanced_sampling and task_type == "regression":
            print("Ignoring --balanced_sampling for regression task.")
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    num_outputs = train_dataset.num_classes if task_type == "classification" else 1
    input_dim = train_dataset.embedding_dim

    model_kwargs = {
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "num_classes": num_outputs,
        "dropout": args.dropout,
    }
    model = DemographicMLP(**model_kwargs).to(device)

    criterion, selection_metric_name = build_training_criterion(
        task_type=task_type,
        train_dataset=train_dataset,
        device=device,
        class_weighted_loss=args.class_weighted_loss,
        selection_metric=args.selection_metric,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"best_mlp_{args.task}.pt"
    best_val_metric = 0.0 if task_type == "classification" else float("inf")

    print(f"\nTraining MLP for {args.task} ({task_type})...")
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters\n")

    for epoch in range(args.epochs):
        train_loss, train_metric = train_epoch(
            model, train_loader, criterion, optimizer, device, task_type=task_type
        )
        val_loss, val_metric = validate(
            model, val_loader, criterion, device, task_type=task_type
        )

        print(f"Epoch {epoch + 1}/{args.epochs}")
        if task_type == "classification":
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metric:.4f}")
            val_selection_metric = val_metric
            if selection_metric_name == "f1_macro":
                val_preds, val_labels = predict(model, val_loader, device, task_type=task_type)
                val_selection_metric = compute_metrics(
                    val_labels, val_preds, val_dataset.num_classes
                )["f1_macro"]
                print(
                    f"  Val Loss: {val_loss:.4f}, Val Acc: {val_metric:.4f}, Val F1(macro): {val_selection_metric:.4f}"
                )
            else:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_metric:.4f}")
            is_better = val_selection_metric > best_val_metric
        else:
            print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_metric:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_metric:.4f}")
            val_selection_metric = val_metric
            is_better = val_metric < best_val_metric

        if is_better:
            best_val_metric = val_selection_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "mlp",
                    "task": args.task,
                    "task_type": task_type,
                    "model_kwargs": model_kwargs,
                    "label_map": train_dataset.label_map,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric_name": selection_metric_name,
                    "val_metric": val_selection_metric,
                },
                checkpoint_path,
            )

    print(f"\nBest validation {selection_metric_name}: {best_val_metric:.4f}")

    if args.evaluate:
        print("\nEvaluating best checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_dataset = EmbeddingDataset(
            args.embedding_dir,
            split=args.test_split,
            task=args.task,
            metadata_csv=args.metadata_csv,
            split_dir=args.split_dir,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        preds, labels = predict(model, test_loader, device, task_type=task_type)
        print("Test Set Results:")
        if task_type == "classification":
            metrics = compute_metrics(labels, preds, test_dataset.num_classes)
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_macro']:.4f}")
        else:
            mae = sum(abs(float(p) - float(y)) for p, y in zip(preds, labels)) / len(labels)
            print(f"  MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
