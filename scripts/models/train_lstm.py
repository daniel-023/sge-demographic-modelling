#!/usr/bin/env python3

"""Train LSTM model for demographic prediction from WavLM embeddings."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.models.architectures.lstm import DemographicLSTM
from scripts.models.training_utils import (
    build_training_criterion,
    get_default_device,
    predict,
    train_epoch,
    validate,
    validate_with_predictions,
)
from scripts.utils.data_loaders import (
    ChunkLengthBatchSampler,
    SequenceEmbeddingDataset,
    pad_sequence_collate,
    task_type_for,
)
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
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, choices=["last", "mean", "attn"], default="mean")
    parser.add_argument("--max_seq_len", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--progress_log_interval",
        type=int,
        default=500,
        help="Print a progress update every N batches during train/val/test passes; 0 disables it.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision on CUDA")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Stop if no improvement after N epochs")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping threshold (e.g., 1.0)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_cached_chunks",
        type=int,
        default=32,
        help="LRU cache size for loaded shard files per worker",
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
    parser.add_argument(
        "--chunk_length_batching",
        dest="chunk_length_batching",
        action="store_true",
        help="Use chunk-local, length-aware batching to reduce shard I/O and padding overhead",
    )
    parser.add_argument(
        "--no_chunk_length_batching",
        dest="chunk_length_batching",
        action="store_false",
        help="Disable chunk-local, length-aware batching",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best model on test split")
    parser.add_argument("--test_split", type=str, default="test", help="Dataset split for evaluation")
    parser.set_defaults(
        class_weighted_loss=True,
        chunk_length_batching=True,
    )
    args = parser.parse_args()

    if args.output_dir is None:
        emb_tag = args.embedding_dir.name
        if args.num_layers == 1:
            emb_tag = f"{emb_tag}_1layer"
        args.output_dir = DEFAULT_RESULTS_ROOT / "lstm" / emb_tag / args.task

    task_type = task_type_for(args.task)
    device = get_default_device()
    print(f"Using device: {device}")

    print("Loading train sequence dataset...")
    train_dataset = SequenceEmbeddingDataset(
        args.embedding_dir,
        split="train",
        task=args.task,
        metadata_csv=args.metadata_csv,
        split_dir=args.split_dir,
        max_seq_len=args.max_seq_len,
        max_cached_chunks=args.max_cached_chunks,
    )
    print(f"Loaded train dataset: {len(train_dataset)} samples")
    print("Loading val sequence dataset...")
    val_dataset = SequenceEmbeddingDataset(
        args.embedding_dir,
        split="val",
        task=args.task,
        metadata_csv=args.metadata_csv,
        split_dir=args.split_dir,
        max_seq_len=args.max_seq_len,
        max_cached_chunks=args.max_cached_chunks,
    )
    print(f"Loaded val dataset: {len(val_dataset)} samples")

    pin_memory = device.type == "cuda"
    train_loader_kwargs = {
        "collate_fn": pad_sequence_collate,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.chunk_length_batching:
        train_loader_kwargs["batch_sampler"] = ChunkLengthBatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
    else:
        train_loader_kwargs["batch_size"] = args.batch_size
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader_kwargs = {
        "collate_fn": pad_sequence_collate,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.chunk_length_batching:
        val_loader_kwargs["batch_sampler"] = ChunkLengthBatchSampler(
            val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
        )
    else:
        val_loader_kwargs["batch_size"] = args.batch_size
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    num_outputs = train_dataset.num_classes if task_type == "classification" else 1
    input_dim = train_dataset.embedding_dim

    model_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_classes": num_outputs,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "use_layer_norm": True,
        "use_residual": False,
        "bidirectional": False,
    }
    model = DemographicLSTM(**model_kwargs).to(device)

    criterion, selection_score_name = build_training_criterion(
        task_type=task_type,
        train_dataset=train_dataset,
        device=device,
        class_weighted_loss=args.class_weighted_loss,
        selection_metric=args.selection_metric,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"best_lstm_{args.task}.pt"
    best_val_score = 0.0 if task_type == "classification" else float("inf")
    epochs_without_improvement = 0

    print(f"\nTraining LSTM for {args.task} ({task_type})...")
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Train samples: {len(train_dataset)} in {len(train_loader)} batches")
    print(f"Val samples: {len(val_dataset)} in {len(val_loader)} batches")
    if args.early_stopping_patience:
        print(f"Early stopping: patience={args.early_stopping_patience}")
    if args.gradient_clip:
        print(f"Gradient clipping: {args.gradient_clip}")
    if args.amp and device.type == "cuda":
        print("AMP: enabled")
    print(f"Chunk+length batching: {'enabled' if args.chunk_length_batching else 'disabled'}")
    print()

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}...")
        train_loss, train_metric = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            task_type=task_type,
            gradient_clip=args.gradient_clip,
            amp=args.amp,
            scaler=scaler,
            progress_interval=args.progress_log_interval,
            progress_prefix=f"train epoch {epoch + 1}",
        )
        if task_type == "classification" and selection_score_name == "f1_macro":
            val_loss, val_score, val_preds, val_labels = validate_with_predictions(
                model,
                val_loader,
                criterion,
                device,
                task_type=task_type,
                amp=args.amp,
                progress_interval=args.progress_log_interval,
                progress_prefix=f"val epoch {epoch + 1}",
            )
        else:
            val_loss, val_score = validate(
                model,
                val_loader,
                criterion,
                device,
                task_type=task_type,
                amp=args.amp,
                progress_interval=args.progress_log_interval,
                progress_prefix=f"val epoch {epoch + 1}",
            )

        print(f"Epoch {epoch + 1}/{args.epochs}")
        if task_type == "classification":
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metric:.4f}")
            selection_score = val_score
            if selection_score_name == "f1_macro":
                selection_score = compute_metrics(
                    val_labels, val_preds, val_dataset.num_classes
                )["f1_macro"]
                print(
                    f"  Val Loss: {val_loss:.4f}, Val Acc: {val_score:.4f}, Val F1(macro): {selection_score:.4f}"
                )
            else:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_score:.4f}")
            is_better = selection_score > best_val_score
        else:
            print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_metric:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_score:.4f}")
            selection_score = val_score
            is_better = val_score < best_val_score

        if is_better:
            best_val_score = selection_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "lstm",
                    "task": args.task,
                    "task_type": task_type,
                    "model_kwargs": model_kwargs,
                    "max_seq_len": args.max_seq_len,
                    "label_map": train_dataset.label_map,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric_name": selection_score_name,
                    "val_metric": selection_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience={args.early_stopping_patience})")
                break

    print(f"\nBest validation {selection_score_name}: {best_val_score:.4f}")

    if args.evaluate:
        print("\nEvaluating best checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_dataset = SequenceEmbeddingDataset(
            args.embedding_dir,
            split=args.test_split,
            task=args.task,
            metadata_csv=args.metadata_csv,
            split_dir=args.split_dir,
            max_seq_len=args.max_seq_len,
            max_cached_chunks=args.max_cached_chunks,
        )
        test_loader_kwargs = {
            "collate_fn": pad_sequence_collate,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        }
        if args.chunk_length_batching:
            test_loader_kwargs["batch_sampler"] = ChunkLengthBatchSampler(
                test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
            )
        else:
            test_loader_kwargs["batch_size"] = args.batch_size
        test_loader = DataLoader(test_dataset, **test_loader_kwargs)

        print(f"Test samples: {len(test_dataset)} in {len(test_loader)} batches")
        preds, labels = predict(model, test_loader, device, task_type=task_type, amp=args.amp,
                                progress_interval=args.progress_log_interval)
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
