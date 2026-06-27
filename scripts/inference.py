#!/usr/bin/env python3
"""
Live inference from a WAV file using a trained MLP or LSTM checkpoint.

The checkpoint is self-describing: it stores the model type, architecture
hyperparameters, task, and label map, so no extra arguments are needed beyond
the checkpoint path and the audio file.

Usage
-----
MLP:
    python -m scripts.inference \
        --checkpoint results/mlp/part1_mlp_full/gender/best_mlp_gender.pt \
        --audio /path/to/speaker.wav

LSTM:
    python -m scripts.inference \
        --checkpoint results/lstm/part1_cap150/ethnicity/best_lstm_ethnicity.pt \
        --audio /path/to/speaker.wav
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torchaudio

from scripts.models.architectures import build_model

TARGET_SR = 16_000
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_audio(path: Path) -> torch.Tensor:
    """Return a mono float32 waveform at 16 kHz, shape [T], on CPU."""
    waveform, sr = torchaudio.load(str(path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # [T]


def _extract_wavlm_frames(waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run WavLM on a CPU waveform and return frame embeddings on device, shape [T, D]."""
    from transformers import AutoFeatureExtractor, AutoModel

    print(f"Loading WavLM: {WAVLM_MODEL_ID}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
    wavlm = AutoModel.from_pretrained(WAVLM_MODEL_ID).to(device)
    wavlm.eval()

    inputs = feature_extractor(
        waveform.numpy(),  # waveform is always CPU from _load_audio
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = wavlm(input_values, output_hidden_states=False)

    return outputs.last_hidden_state.squeeze(0)  # [T, D] on device


def _print_result(task: str, task_type: str, label_map: dict | None, logits: torch.Tensor) -> None:
    if task_type == "regression":
        print(f"\nPredicted {task}: {logits.squeeze().item():.2f}")
        return

    probs = torch.softmax(logits.squeeze(0), dim=-1)
    idx_to_label: dict[int, str] = {}
    if label_map:
        idx_to_label = {int(v): str(k) for k, v in label_map.items()}

    sorted_probs, sorted_indices = probs.sort(descending=True)
    print(f"\nPredicted {task}:")
    for rank, (idx, prob) in enumerate(zip(sorted_indices.tolist(), sorted_probs.tolist()), start=1):
        label = idx_to_label.get(idx, str(idx))
        marker = " <--" if rank == 1 else ""
        print(f"  {rank}. {label:15s}  {prob * 100:5.1f}%{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Run live demographic inference on a WAV file using a trained checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained .pt checkpoint (MLP or LSTM).")
    parser.add_argument("--audio", type=Path, required=True, help="Path to a WAV audio file for inference.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")

    device = _detect_device()
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type")
    task = checkpoint.get("task")
    task_type = checkpoint.get("task_type", "classification")
    model_kwargs = checkpoint.get("model_kwargs")
    label_map = checkpoint.get("label_map")
    max_seq_len = checkpoint.get("max_seq_len")

    if not model_type or not task or not model_kwargs:
        sys.exit(
            "Checkpoint is missing required metadata (model_type / task / model_kwargs). "
            "Re-train with updated scripts."
        )

    epoch = checkpoint.get("epoch")
    val_metric = checkpoint.get("val_metric")
    val_metric_name = checkpoint.get("val_metric_name", "metric")
    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"  model={model_type.upper()}, task={task}, type={task_type}")
    if epoch is not None:
        print(f"  trained for {epoch + 1} epoch(s)", end="")
        if val_metric is not None:
            print(f", best val {val_metric_name}={val_metric:.4f}", end="")
        print()

    model = build_model(model_type, model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"\nAudio: {args.audio}")
    waveform = _load_audio(args.audio)
    duration_s = waveform.shape[0] / TARGET_SR
    print(f"  duration={duration_s:.2f}s, samples={waveform.shape[0]}")

    frames = _extract_wavlm_frames(waveform, device)  # [T, D] on device
    print(f"  WavLM frames: T={frames.shape[0]}, D={frames.shape[1]}")

    with torch.no_grad():
        if model_type == "mlp":
            logits = model(frames.mean(dim=0, keepdim=True))  # [1, D] → logits
        else:
            seq = frames[:max_seq_len].unsqueeze(0) if max_seq_len else frames.unsqueeze(0)  # [1, T, D]
            lengths = torch.tensor([seq.shape[1]], dtype=torch.long, device=device)
            logits = model(seq, lengths=lengths)

    _print_result(task, task_type, label_map, logits.cpu())


if __name__ == "__main__":
    main()
