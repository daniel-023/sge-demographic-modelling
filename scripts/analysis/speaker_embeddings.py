import argparse
from pathlib import Path
from collections import defaultdict
import torch
import pandas as pd
from tqdm import tqdm


def age_bin(age: int) -> str:
    decade = age // 10
    return f"{decade}X+" if decade >= 7 else f"{decade}X"


def normalize_age(age):
    if age is None or pd.isna(age):
        return None
    if isinstance(age, str):
        s = age.strip()
        if s == "7X":
            return "7X+"
        if s.endswith("X") and s[:-1].isdigit():  # already "2X", "3X", etc.
            return s
        if s.isdigit():
            return age_bin(int(s))
        return None
    return age_bin(int(age))


def aggregate_speaker_embeddings_from_mlp_shards(mlp_shard_dir: Path) -> dict:
    mlp_shard_dir = Path(mlp_shard_dir)
    shard_files = sorted(mlp_shard_dir.glob("mlp_shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No mlp_shard_*.pt files found in {mlp_shard_dir}")

    speaker_sums = defaultdict(lambda: {"sum": None, "count": 0})

    for shard_path in tqdm(shard_files, desc="Processing MLP shards"):
        shard = torch.load(shard_path, map_location="cpu")
        embs = shard["embeddings"]
        spks = shard["speaker_ids"]

        for spk, emb in zip(spks, embs):
            spk = str(spk).strip().zfill(4)     # normalize to XXX format
            emb32 = emb.to(torch.float32)
            if speaker_sums[spk]["sum"] is None:
                speaker_sums[spk]["sum"] = emb32.clone()
            else:
                speaker_sums[spk]["sum"] += emb32
            speaker_sums[spk]["count"] += 1

        del shard

    return {spk: d["sum"] / d["count"] for spk, d in speaker_sums.items() if d["count"] > 0}


def main():
    p = argparse.ArgumentParser(description="Compute speaker-level mean embeddings from MLP shards for UMAP.")
    p.add_argument("--mlp-shard-dir", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--out-path", type=Path, required=True)
    args = p.parse_args()

    print("[1/3] Computing speaker embeddings from MLP shards...")
    speaker_embeddings = aggregate_speaker_embeddings_from_mlp_shards(args.mlp_shard_dir)
    print(f"Computed embeddings for {len(speaker_embeddings)} speakers")

    print("[2/3] Loading speaker metadata...")
    metadata_df = pd.read_csv(args.metadata_csv, dtype={"speaker_id": str})
    if "speaker_id" not in metadata_df.columns:
        raise ValueError("Metadata CSV must contain a 'speaker_id' column")

    metadata_df["speaker_id"] = metadata_df["speaker_id"].str.strip().str.zfill(4)  # normalize here
    metadata_df.set_index("speaker_id", inplace=True)

    print("[3/3] Matching embeddings with metadata and saving...")
    rows = []
    matched = 0
    missing = 0

    for spk, emb in speaker_embeddings.items():
        if spk not in metadata_df.index:
            missing += 1
            continue
        meta = metadata_df.loc[spk]

        rows.append({
            "speaker_id": spk,
            "embedding": emb.numpy(),
            "age": normalize_age(meta["age"]) if "age" in meta else None,
            "gender": meta["gender"] if "gender" in meta else None,
            "ethnicity": meta["ethnicity"] if "ethnicity" in meta else None,
        })
        matched += 1

    print(f"Matched {matched} speakers, {missing} missing from metadata")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, args.out_path)
    print(f"Saved speaker embeddings → {args.out_path}")


if __name__ == "__main__":
    main()