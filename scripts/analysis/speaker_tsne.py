#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def label_order_and_colors(category: str, unique_labels: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return a stable plotting order and category-appropriate colors."""
    preferred_orders = {
        "gender": ["F", "M", "UNKNOWN"],
        "ethnicity": ["CHINESE", "MALAY", "INDIAN", "OTHER", "UNKNOWN"],
        "age_group": ["1X", "2X", "3X", "4X", "5X", "6X", "7X+", "UNKNOWN"],
    }
    preferred_colors = {
        "gender": {
            "F": "#0072B2",
            "M": "#D55E00",
            "UNKNOWN": "#8f8f8f",
        },
        "ethnicity": {
            "CHINESE": "#0072B2",
            "MALAY": "#E69F00",
            "INDIAN": "#009E73",
            "OTHER": "#CC79A7",
            "UNKNOWN": "#8f8f8f",
        },
        "age_group": {
            "1X": "#355070",
            "2X": "#6D597A",
            "3X": "#43AA8B",
            "4X": "#90BE6D",
            "5X": "#F9C74F",
            "6X": "#F8961E",
            "7X+": "#F94144",
            "UNKNOWN": "#8f8f8f",
        },
    }

    preferred_order = preferred_orders.get(category, [])
    ordered = [label for label in preferred_order if label in unique_labels]
    ordered.extend(sorted(label for label in unique_labels if label not in ordered))

    category_colors = preferred_colors.get(category, {})
    fallback_palette = sns.color_palette("colorblind", len(ordered))
    color_map = {
        label: category_colors.get(label, fallback_palette[i])
        for i, label in enumerate(ordered)
    }
    return ordered, color_map


def load_speaker_embeddings(emb_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load speaker embeddings and metadata."""
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    df = pd.DataFrame(
        {
            "speaker_id": [d.get("speaker_id") for d in data],
            "speaker_key": [d.get("speaker_key") for d in data],
            "part": [d.get("part") for d in data],
            "age": [d.get("age") for d in data],
            "gender": [d.get("gender") for d in data],
            "ethnicity": [d.get("ethnicity") for d in data],
        }
    )
    embeddings = np.asarray([d["embedding"] for d in data], dtype=np.float32)
    return embeddings, df


def preprocess_embeddings(
    embeddings: np.ndarray,
    l2_normalize: bool,
    standardize: bool,
) -> np.ndarray:
    """Optionally normalize embeddings before t-SNE."""
    x = embeddings.astype(np.float32, copy=True)

    if l2_normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        x = x / norms

    if standardize:
        x = StandardScaler().fit_transform(x)

    return x


def apply_tsne(
    embeddings: np.ndarray,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    init: str,
    metric: str,
) -> np.ndarray:
    """Fit t-SNE and return 2D coordinates."""
    tsne_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init=init,
        metric=metric,
        random_state=42,
        verbose=1,
    )
    return tsne_model.fit_transform(embeddings)


def plot_tsne(
    coords: np.ndarray,
    labels: pd.Series,
    out_path: Path,
    title: str,
    category: str,
) -> None:
    """Create and save t-SNE scatter plot for a given label series."""
    valid_mask = labels.notna()
    coords = coords[valid_mask.to_numpy()]
    labels = labels[valid_mask].astype(str)

    plt.figure(figsize=(13, 10))
    ax = plt.gca()
    ax.set_facecolor("white")

    unique_labels = sorted(labels.unique())
    ordered_labels, color_map = label_order_and_colors(category, unique_labels)
    legend_ncol = len(ordered_labels) if category == "age_group" else min(len(ordered_labels), 4)

    for label in ordered_labels:
        mask = labels == label
        plt.scatter(
            coords[mask.to_numpy(), 0],
            coords[mask.to_numpy(), 1],
            c=[color_map[label]],
            label=label,
            alpha=0.7,
            s=42,
            edgecolors="none",
            linewidth=0.0,
        )

    plt.xlabel("t-SNE 1", fontsize=24)
    plt.ylabel("t-SNE 2", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=28, fontweight="bold")
    ax.grid(False)
    sns.despine()
    plt.legend(
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        ncol=legend_ncol,
        frameon=True,
        fontsize=18,
        title_fontsize=20,
        markerscale=1.9,
        facecolor="white",
        edgecolor="#d0cbc2",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved plot -> {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE visualization for speaker embeddings")
    parser.add_argument("--emb-path", type=Path, required=True, help="Path to speaker_embeddings.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for plots/coordinates")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--learning-rate", type=float, default=200.0, help="t-SNE learning rate")
    parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE optimization iterations")
    parser.add_argument(
        "--init",
        type=str,
        default="pca",
        choices=["pca", "random"],
        help="t-SNE initialization strategy",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance metric for t-SNE",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable row-wise L2 normalization before t-SNE",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization before t-SNE",
    )
    args = parser.parse_args()

    print("[1/5] Loading speaker embeddings...")
    raw_embeddings, df = load_speaker_embeddings(args.emb_path)
    print(f"Loaded {len(df)} speakers, embedding dim={raw_embeddings.shape[1]}")

    print("[2/5] Preparing labels...")
    df["age_group"] = df["age"]
    part_name = args.out_dir.name
    if part_name == "part1":
        part_label = "p1"
    elif part_name == "part2":
        part_label = "p2"
    elif part_name == "part3":
        part_label = "p3"
    elif part_name == "parts1_3":
        part_label = "p1_3"
    else:
        part_label = part_name

    print("[3/5] Preprocessing embeddings...")
    embeddings = preprocess_embeddings(
        raw_embeddings,
        l2_normalize=not args.no_l2_normalize,
        standardize=not args.no_standardize,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[4/5] Computing t-SNE...")
    max_perplexity = max(1.0, float(len(embeddings) - 1))
    perplexity = min(args.perplexity, max_perplexity)
    if perplexity != args.perplexity:
        print(f"[!] Adjusted perplexity from {args.perplexity} to {perplexity} for sample size")
    base_coords = apply_tsne(
        embeddings,
        perplexity=perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init=args.init,
        metric=args.metric,
    )
    plot_specs = [
        ("gender", "Gender", "gender"),
        ("ethnicity", "Ethnicity", "ethnicity"),
        ("age_group", "Age", "age"),
    ]
    for col, title, stem in plot_specs:
        plot_tsne(
            base_coords,
            df[col],
            args.out_dir / f"tsne_{stem}_{part_label}.png",
            title,
            col,
        )

    print("[5/5] Saving coordinates...")
    coords_df = df.copy()
    coords_df["tsne_1"] = base_coords[:, 0]
    coords_df["tsne_2"] = base_coords[:, 1]
    coords_df.to_csv(args.out_dir / "speaker_tsne_coords_unsupervised.csv", index=False)
    print(f"[+] Saved coordinates -> {args.out_dir / 'speaker_tsne_coords_unsupervised.csv'}")

    print("[+] t-SNE visualization complete")


if __name__ == "__main__":
    main()
