#!/usr/bin/env python3
"""
Dual-purpose WavLM extraction for:
  (A) MLP: utterance-level pooled embeddings  [D]
  (B) LSTM: variable-length frame sequences  [T, D]  (TRIMMED; no padding saved)

Edits vs your version:
  1) Use AutoFeatureExtractor (safer for WavLM configs)
  2) Cache torchaudio resamplers (much faster)
  3) Make last_hidden_state access robust under DataParallel
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
from pathlib import Path
from typing import List, Dict
import warnings

import torch
import torchaudio
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm

TARGET_SR = 16000
MODEL_ID = "microsoft/wavlm-base-plus"

DEFAULT_BATCH_SIZE = 64
DEFAULT_MLP_SHARD_UTTS = 5000
DEFAULT_LSTM_SHARD_UTTS = 256
DEFAULT_MAX_FRAMES = 250  # set None / 0 to disable
DEFAULT_LSTM_MAX_UTTS_PER_SPEAKER = 150

# Cache resamplers by source SR
_RESAMPLERS: Dict[int, torchaudio.transforms.Resample] = {}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract WavLM embeddings for MLP + LSTM (sharded, streamable).")
    p.add_argument(
        "--part",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Part ID used for default paths when --utt-csv/--out-dir are not provided.",
    )
    p.add_argument(
        "--utt-csv",
        type=Path,
        default=None,
        help="Utterance CSV with path, utt_id, speaker_id. Default: data/metadata/cleaned/part{part}_utterances.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory root. Default: results/embeddings/baseplus/part{part}",
    )
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test"], help="Optional split filter.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Extraction batch size.")
    p.add_argument("--mlp-shard-utts", type=int, default=DEFAULT_MLP_SHARD_UTTS, help="Utterances per MLP shard.")
    p.add_argument("--lstm-shard-utts", type=int, default=DEFAULT_LSTM_SHARD_UTTS, help="Utterances per LSTM shard.")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Optional cap on frames per utterance for LSTM.")
    p.add_argument(
        "--max-utts-per-speaker",
        type=int,
        default=None,
        help="Deprecated global cap applied to both outputs unless a per-output cap is provided.",
    )
    p.add_argument(
        "--mlp-max-utts-per-speaker",
        type=int,
        default=None,
        help="Optional per-speaker cap applied only to MLP embeddings.",
    )
    p.add_argument(
        "--lstm-max-utts-per-speaker",
        type=int,
        default=DEFAULT_LSTM_MAX_UTTS_PER_SPEAKER,
        help="Per-speaker cap applied only to LSTM embeddings.",
    )
    p.add_argument(
        "--utt-sample-seed",
        type=int,
        default=42,
        help="Random seed for per-speaker utterance sampling when --max-utts-per-speaker is set.",
    )
    p.add_argument(
        "--outputs",
        type=str,
        default="both",
        choices=["both", "mlp", "lstm"],
        help="Which embeddings to write.",
    )
    return p.parse_args()


def sample_per_speaker_indices(df: pd.DataFrame, cap: int, seed: int) -> set[int]:
    if cap <= 0:
        raise ValueError("--max-utts-per-speaker must be a positive integer")

    speaker_to_indices = df.groupby("speaker_id").indices
    rng = random.Random(seed)
    keep_indices = []
    for speaker_id in sorted(speaker_to_indices.keys()):
        indices = list(speaker_to_indices[speaker_id])
        if len(indices) <= cap:
            keep_indices.extend(indices)
        else:
            keep_indices.extend(rng.sample(indices, cap))

    return set(keep_indices)


def load_wav(path: str) -> torch.Tensor:
    """Load wav -> mono -> resample -> return 1D float tensor on CPU."""
    wav, sr = torchaudio.load(path)  # [C, N]
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # [N]

    if sr != TARGET_SR:
        if sr not in _RESAMPLERS:
            _RESAMPLERS[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
        wav = _RESAMPLERS[sr](wav)

    return wav


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def next_shard_id(shard_dir: Path, prefix: str) -> int:
    """Resume-friendly: infer next shard id from existing files."""
    existing = sorted(shard_dir.glob(f"{prefix}_shard_*.pt"))
    if not existing:
        return 0
    last = existing[-1].stem
    n = int(last.split("_shard_")[-1])
    return n + 1


def flush_mlp_shard(
    out_dir: Path,
    shard_id: int,
    emb_buf: List[torch.Tensor],
    utt_buf: List[str],
    spk_buf: List[str],
) -> None:
    if not emb_buf:
        return
    payload = {
        "embeddings": torch.stack(emb_buf, dim=0),  # [N, D]
        "utt_ids": utt_buf,
        "speaker_ids": spk_buf,
    }
    out_path = out_dir / f"mlp_shard_{shard_id:06d}.pt"
    torch.save(payload, out_path)


def flush_lstm_shard(
    out_dir: Path,
    shard_id: int,
    seq_buf: List[torch.Tensor],
    len_buf: List[int],
    utt_buf: List[str],
    spk_buf: List[str],
) -> None:
    if not seq_buf:
        return
    if not (len(seq_buf) == len(len_buf) == len(utt_buf) == len(spk_buf)):
        raise ValueError(
            "LSTM shard buffer length mismatch: "
            f"sequences={len(seq_buf)}, lengths={len(len_buf)}, "
            f"utt_ids={len(utt_buf)}, speaker_ids={len(spk_buf)}"
        )
    payload = {
        "sequences": seq_buf,                # list of [T_i, D] CPU tensors (trimmed)
        "lengths": torch.tensor(len_buf, dtype=torch.int32),  # [N]
        "utt_ids": utt_buf,
        "speaker_ids": spk_buf,
    }
    out_path = out_dir / f"lstm_shard_{shard_id:06d}.pt"
    torch.save(payload, out_path)
    length_tensor = payload["lengths"]
    print(
        f"[LSTM SHARD] {out_path.name} | N={len(seq_buf)} | "
        f"len(min/med/max)=({int(length_tensor.min().item())}/"
        f"{int(length_tensor.median().item())}/"
        f"{int(length_tensor.max().item())})"
    )


def main():
    args = parse_args()
    do_mlp = args.outputs in ("both", "mlp")
    do_lstm = args.outputs in ("both", "lstm")

    warnings.filterwarnings(
        "ignore",
        message="Support for mismatched key_padding_mask and attn_mask is deprecated.*",
        category=UserWarning,
    )

    root = repo_root()
    if args.utt_csv is None:
        args.utt_csv = root / "data" / "metadata" / "utterances" / f"part{args.part}_utterances.csv"
    if args.out_dir is None:
        args.out_dir = root / "results" / "embeddings" / "baseplus" / f"part{args.part}"

    # Output dirs
    out_root = args.out_dir / args.split if args.split else args.out_dir
    mlp_dir = out_root / "mlp_shards"
    lstm_dir = out_root / "lstm_shards"
    if do_mlp:
        safe_mkdir(mlp_dir)
    if do_lstm:
        safe_mkdir(lstm_dir)

    # Resume-friendly shard counters
    mlp_shard_id = next_shard_id(mlp_dir, "mlp") if do_mlp else 0
    lstm_shard_id = next_shard_id(lstm_dir, "lstm") if do_lstm else 0

    # Load manifest
    print(f"[CONFIG] part={args.part}")
    print(f"[CONFIG] utt_csv={args.utt_csv}")
    print(f"[CONFIG] out_dir={out_root}")
    if args.max_utts_per_speaker is not None:
        print(
            f"[CONFIG] global_max_utts_per_speaker={args.max_utts_per_speaker} "
            f"(seed={args.utt_sample_seed})"
        )
    if do_mlp and args.mlp_max_utts_per_speaker is not None:
        print(
            f"[CONFIG] mlp_max_utts_per_speaker={args.mlp_max_utts_per_speaker} "
            f"(seed={args.utt_sample_seed})"
        )
    if do_lstm and args.lstm_max_utts_per_speaker is not None:
        print(
            f"[CONFIG] lstm_max_utts_per_speaker={args.lstm_max_utts_per_speaker} "
            f"(seed={args.utt_sample_seed})"
        )

    df = pd.read_csv(args.utt_csv)
    required = {"path", "utt_id", "speaker_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.utt_csv}: {sorted(missing)}")

    if args.split:
        if "split" not in df.columns:
            raise ValueError("You passed --split but CSV has no 'split' column.")
        df = df[df["split"] == args.split]

    df = df[df["path"].notna()].reset_index(drop=True)
    original_rows = len(df)

    mlp_cap = args.mlp_max_utts_per_speaker
    lstm_cap = args.lstm_max_utts_per_speaker
    if mlp_cap is not None and mlp_cap <= 0:
        mlp_cap = None
    if lstm_cap is not None and lstm_cap <= 0:
        lstm_cap = None
    if args.max_utts_per_speaker is not None:
        if mlp_cap is None:
            mlp_cap = args.max_utts_per_speaker
        if lstm_cap is None:
            lstm_cap = args.max_utts_per_speaker

    mlp_keep_indices = set(range(len(df)))
    lstm_keep_indices = set(range(len(df)))
    if do_mlp and mlp_cap is not None:
        mlp_keep_indices = sample_per_speaker_indices(df, mlp_cap, args.utt_sample_seed)
        print(f"[CONFIG] mlp_rows_after_cap={len(mlp_keep_indices)} (from {original_rows})")
    if do_lstm and lstm_cap is not None:
        lstm_keep_indices = sample_per_speaker_indices(df, lstm_cap, args.utt_sample_seed)
        print(f"[CONFIG] lstm_rows_after_cap={len(lstm_keep_indices)} (from {original_rows})")

    # Model setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)

    # Optional: DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Shard buffers
    mlp_emb_buf: List[torch.Tensor] = []
    mlp_utt_buf: List[str] = []
    mlp_spk_buf: List[str] = []

    lstm_seq_buf: List[torch.Tensor] = []
    lstm_len_buf: List[int] = []
    lstm_utt_buf: List[str] = []
    lstm_spk_buf: List[str] = []

    failed = 0

    for i in tqdm(range(0, len(df), args.batch_size), desc="Extracting WavLM (dual)"):
        batch = df.iloc[i : i + args.batch_size]

        batch_start = i
        # Faster than iterrows for ids
        paths = batch["path"].tolist()
        utt_ids_all = batch["utt_id"].astype(str).tolist()
        spk_ids_all = batch["speaker_id"].astype(str).tolist()

        waveforms = []
        utt_ids = []
        speaker_ids = []

        # Load audio
        row_indices = []
        for offset, (pth, utt, spk) in enumerate(zip(paths, utt_ids_all, spk_ids_all)):
            try:
                wav = load_wav(str(pth))
                waveforms.append(wav.numpy())  # feature_extractor expects numpy/list
                utt_ids.append(utt)
                speaker_ids.append(spk)
                row_indices.append(batch_start + offset)
            except Exception as e:
                print(f"[WARN] Failed loading {pth}: {e}")
                failed += 1

        if not waveforms:
            continue

        # Featurize + forward
        inputs = feature_extractor(
            waveforms,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            # Robust under DataParallel (sometimes returns tuple)
            hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]  # [B, T, D]
            attn = inputs["attention_mask"]  # [B, T_audio]
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            feat_mask = base_model._get_feature_vector_attention_mask(hs.shape[1], attn)  # [B, T_feat]
            
            pooled_cpu = None
            if do_mlp:
                # ----- MLP pooled embeddings: masked mean pooling -----
                mask = feat_mask.unsqueeze(-1).to(hs.dtype)  # [B, T_feat, 1]
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled_cpu = pooled.cpu()

            hs_cpu = hs.cpu() if do_lstm else None
            feat_lengths = feat_mask.sum(dim=1).to("cpu") if do_lstm else None

        # Add to buffers utterance-by-utterance
        B = hs.size(0)
        for b in range(B):
            utt = utt_ids[b]
            spk = speaker_ids[b]
            row_idx = row_indices[b]

            if do_mlp and pooled_cpu is not None and row_idx in mlp_keep_indices:
                # MLP
                mlp_emb_buf.append(pooled_cpu[b].contiguous())
                mlp_utt_buf.append(utt)
                mlp_spk_buf.append(spk)

            if do_lstm and hs_cpu is not None and feat_lengths is not None and row_idx in lstm_keep_indices:
                # LSTM
                L = int(feat_lengths[b].item())
                if args.max_frames and args.max_frames > 0:
                    L = min(L, args.max_frames)
                L = max(1, min(L, hs_cpu.shape[1]))

                # Clone after slicing so each saved sequence owns only its
                # trimmed data instead of retaining the larger batch storage.
                seq = hs_cpu[b, :L].clone()
                lstm_seq_buf.append(seq)
                lstm_len_buf.append(L)
                lstm_utt_buf.append(utt)
                lstm_spk_buf.append(spk)

        # Flush shards when buffers reach capacity
        if do_mlp and len(mlp_emb_buf) >= args.mlp_shard_utts:
            flush_mlp_shard(mlp_dir, mlp_shard_id, mlp_emb_buf, mlp_utt_buf, mlp_spk_buf)
            mlp_shard_id += 1
            mlp_emb_buf.clear(); mlp_utt_buf.clear(); mlp_spk_buf.clear()

        if do_lstm and len(lstm_seq_buf) >= args.lstm_shard_utts:
            flush_lstm_shard(lstm_dir, lstm_shard_id, lstm_seq_buf, lstm_len_buf, lstm_utt_buf, lstm_spk_buf)
            lstm_shard_id += 1
            lstm_seq_buf.clear(); lstm_len_buf.clear(); lstm_utt_buf.clear(); lstm_spk_buf.clear()

    # Final flush
    if do_mlp and mlp_emb_buf:
        flush_mlp_shard(mlp_dir, mlp_shard_id, mlp_emb_buf, mlp_utt_buf, mlp_spk_buf)

    if do_lstm and lstm_seq_buf:
        flush_lstm_shard(lstm_dir, lstm_shard_id, lstm_seq_buf, lstm_len_buf, lstm_utt_buf, lstm_spk_buf)

    print(f"[DONE] Extraction complete. Failed utterances: {failed}")
    if do_mlp:
        print(f"MLP shards in:  {mlp_dir}")
    if do_lstm:
        print(f"LSTM shards in: {lstm_dir}")


if __name__ == "__main__":
    main()
