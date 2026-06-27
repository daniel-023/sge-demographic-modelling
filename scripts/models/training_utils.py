#!/usr/bin/env python3

"""Shared training and evaluation helpers for model scripts."""

from contextlib import nullcontext
import time

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(dataset, num_classes=None):
    labels = [int(sample[-1]) for sample in dataset.samples]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if num_classes is None:
        num_classes = int(labels_tensor.max().item()) + 1

    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels_tensor]

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

def _unpack_batch(batch):
    if len(batch) == 2:
        inputs, labels = batch
        return inputs, labels, None
    if len(batch) == 3:
        inputs, labels, lengths = batch
        return inputs, labels, lengths
    raise ValueError(f"Unsupported batch format with {len(batch)} elements")


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    task_type="classification",
    gradient_clip=None,
    amp=False,
    scaler=None,
    progress_interval=0,
    progress_prefix="train",
):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    total_error = 0.0
    use_amp = bool(amp and device.type == "cuda")
    non_blocking = device.type == "cuda"
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader, start=1):
        inputs, labels, lengths = _unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        lengths = lengths.to(device, non_blocking=non_blocking) if lengths is not None else None

        optimizer.zero_grad()
        amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
        with amp_ctx():
            logits = model(inputs, lengths=lengths) if lengths is not None else model(inputs)

            if task_type == "regression":
                labels = labels.float().unsqueeze(1)
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)

        if use_amp:
            if scaler is None:
                scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        if task_type == "regression":
            total_error += torch.sum((logits - labels).abs()).item()
        else:
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

        total_loss += loss.item()
        total += labels.size(0)

        if progress_interval and (
            batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(dataloader)
        ):
            elapsed = time.perf_counter() - epoch_start
            print(
                f"[{progress_prefix}] batch {batch_idx}/{len(dataloader)} "
                f"elapsed={elapsed:.1f}s"
            )

    metric = (total_error / total) if task_type == "regression" else (correct / total)
    return total_loss / len(dataloader), metric


def validate(
    model,
    dataloader,
    criterion,
    device,
    task_type="classification",
    amp=False,
    progress_interval=0,
    progress_prefix="val",
):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    total_error = 0.0
    use_amp = bool(amp and device.type == "cuda")
    non_blocking = device.type == "cuda"
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    val_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            inputs, labels, lengths = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking) if lengths is not None else None

            with amp_ctx():
                logits = model(inputs, lengths=lengths) if lengths is not None else model(inputs)

                if task_type == "regression":
                    labels = labels.float().unsqueeze(1)
                    loss = criterion(logits, labels)
                else:
                    loss = criterion(logits, labels)

            if task_type == "regression":
                total_error += torch.sum((logits - labels).abs()).item()
            else:
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()

            total_loss += loss.item()
            total += labels.size(0)

            if progress_interval and (
                batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(dataloader)
            ):
                elapsed = time.perf_counter() - val_start
                print(
                    f"[{progress_prefix}] batch {batch_idx}/{len(dataloader)} "
                    f"elapsed={elapsed:.1f}s"
                )

    metric = (total_error / total) if task_type == "regression" else (correct / total)
    return total_loss / len(dataloader), metric


def validate_with_predictions(
    model,
    dataloader,
    criterion,
    device,
    task_type="classification",
    amp=False,
    progress_interval=0,
    progress_prefix="val",
):
    """Run a single validation pass and also return predictions/labels."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    total_error = 0.0
    all_preds = []
    all_labels = []
    use_amp = bool(amp and device.type == "cuda")
    non_blocking = device.type == "cuda"
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    val_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            inputs, labels, lengths = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking) if lengths is not None else None

            with amp_ctx():
                logits = model(inputs, lengths=lengths) if lengths is not None else model(inputs)

                if task_type == "regression":
                    labels_for_loss = labels.float().unsqueeze(1)
                    loss = criterion(logits, labels_for_loss)
                else:
                    labels_for_loss = labels
                    loss = criterion(logits, labels_for_loss)

            if task_type == "regression":
                preds = logits.squeeze(1)
                total_error += torch.sum((preds - labels_for_loss.squeeze(1)).abs()).item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels_for_loss.squeeze(1).cpu().tolist())
            else:
                _, predicted = logits.max(1)
                correct += predicted.eq(labels_for_loss).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels_for_loss.cpu().tolist())

            total_loss += loss.item()
            total += labels_for_loss.size(0)

            if progress_interval and (
                batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(dataloader)
            ):
                elapsed = time.perf_counter() - val_start
                print(
                    f"[{progress_prefix}] batch {batch_idx}/{len(dataloader)} "
                    f"elapsed={elapsed:.1f}s"
                )

    metric = (total_error / total) if task_type == "regression" else (correct / total)
    return total_loss / len(dataloader), metric, all_preds, all_labels


def predict(
    model,
    dataloader,
    device,
    task_type="classification",
    amp=False,
    progress_interval=0,
    progress_prefix="predict",
):
    model.eval()
    all_preds = []
    all_labels = []
    use_amp = bool(amp and device.type == "cuda")
    non_blocking = device.type == "cuda"
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    predict_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            inputs, labels, lengths = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking) if lengths is not None else None

            with amp_ctx():
                logits = model(inputs, lengths=lengths) if lengths is not None else model(inputs)
            if task_type == "regression":
                preds = logits.squeeze(1).cpu().tolist()
            else:
                _, predicted = logits.max(1)
                preds = predicted.cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            if progress_interval and (
                batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(dataloader)
            ):
                elapsed = time.perf_counter() - predict_start
                print(
                    f"[{progress_prefix}] batch {batch_idx}/{len(dataloader)} "
                    f"elapsed={elapsed:.1f}s"
                )

    return all_preds, all_labels
