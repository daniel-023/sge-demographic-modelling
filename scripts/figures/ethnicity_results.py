#!/usr/bin/env python3

"""Generate README figures from saved ethnicity evaluation metrics."""

import json
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "sge_demographic_figures_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "fontconfig"))

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = REPO_ROOT / "results" / "evaluation"
OUTPUT_DIR = REPO_ROOT / "figures"
PARTS = (1, 2, 3)
MODELS = ("mlp", "lstm")
ETHNICITY_ORDER = ("CHINESE", "MALAY", "INDIAN")
ETHNICITY_LABELS = ("Chinese", "Malay", "Indian")
AGE_GROUPS = ("1X", "2X", "3X", "4X", "5X", "6X")
MODEL_LABELS = {"mlp": "MLP", "lstm": "LSTM"}
MODEL_COLORS = {"mlp": "#737373", "lstm": "#2E86C1"}
FIG_DPI = 220


def load_ethnicity_metrics(part, model):
    path = METRICS_ROOT / f"part{part}" / f"{model}_ethnicity_test_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ethnicity metrics: {path}. Run evaluation before generating figures."
        )
    return json.loads(path.read_text())


def percent(value):
    return float(value) * 100.0


def plot_per_class_f1():
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), dpi=FIG_DPI, sharey=True)
    width = 0.36
    x = np.arange(len(ETHNICITY_ORDER))

    for part, ax in zip(PARTS, axes):
        for offset, model in [(-width / 2, "mlp"), (width / 2, "lstm")]:
            metrics = load_ethnicity_metrics(part, model)
            values = [
                percent(metrics["per_class_metrics"][label]["f1"])
                for label in ETHNICITY_ORDER
            ]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
            ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=9)

        ax.set_title(f"Part {part}", fontsize=13)
        ax.set_xticks(x, ETHNICITY_LABELS)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("F1 (%)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    output_path = OUTPUT_DIR / "ethnicity_per_class_f1.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_age_group_performance():
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.2), dpi=FIG_DPI, sharex=True)
    x = np.arange(len(AGE_GROUPS))

    for col, part in enumerate(PARTS):
        for row, metric_name in enumerate(("accuracy", "f1_macro")):
            ax = axes[row, col]
            for model, linestyle, marker in [
                ("mlp", "--", "o"),
                ("lstm", "-", "s"),
            ]:
                metrics = load_ethnicity_metrics(part, model)
                subgroup = metrics["subgroup_metrics"]["age_bin"]
                values = [percent(subgroup[group][metric_name]) for group in AGE_GROUPS]
                ax.plot(
                    x,
                    values,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.2,
                    markersize=5,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )

            if row == 0:
                ax.set_title(f"Part {part}", fontsize=13)
                ax.set_ylim(20, 100)
                ax.set_yticks([20, 40, 60, 80, 100])
            else:
                ax.set_ylim(20, 90)
                ax.set_yticks([20, 30, 40, 50, 60, 70, 80, 90])
                ax.set_xlabel("Age group", fontsize=12)
            ax.set_xticks(x, AGE_GROUPS)
            ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
            ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1, 0].set_ylabel("Macro F1 (%)", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = OUTPUT_DIR / "ethnicity_by_age_group.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    plot_per_class_f1()
    plot_age_group_performance()


if __name__ == "__main__":
    main()
