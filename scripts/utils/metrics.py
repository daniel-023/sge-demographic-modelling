#!/usr/bin/env python3

"""
Evaluation metrics and visualization utilities.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, num_classes):
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = f1
    
    return metrics


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    labels=None,
    output_path=None,
    figsize=(8, 6),
    normalize=None,
    annotate_counts=True,
    annotate_percentages=True,
    percentage_symbol=False,
    annotation_decimals=1,
    show_title=True,
    title_text=None,
    show_colorbar=True,
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names for labels
        output_path: Path to save figure
        figsize: Figure size
        normalize: One of {None, "true", "pred", "all"} for matrix normalization
        annotate_counts: Whether to include raw counts in cell annotation
        annotate_percentages: Whether to include percentages in cell annotation
        percentage_symbol: Whether to append '%' to percentage annotations
        annotation_decimals: Decimal places in numeric annotations
        show_title: Whether to show the chart title
        title_text: Optional custom title text
        show_colorbar: Whether to show the colorbar
    """
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize is None:
        cm_plot = cm_counts
        pct_matrix = None
    else:
        cm_plot = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize) * 100.0
        pct_matrix = cm_plot
    
    plt.figure(figsize=figsize)
    heatmap_kwargs = {
        "cmap": "Blues",
        "cbar": show_colorbar,
        "square": True,
        "linewidths": 0.5,
        "linecolor": "white",
        "annot_kws": {"fontsize": 13},
    }
    if normalize == "true":
        heatmap_kwargs["vmin"] = 0.0
        heatmap_kwargs["vmax"] = 100.0
    if class_names is not None:
        if len(class_names) != cm_counts.shape[0]:
            raise ValueError(
                f"class_names length ({len(class_names)}) must match confusion-matrix size ({cm_counts.shape[0]})."
            )
        heatmap_kwargs["xticklabels"] = class_names
        heatmap_kwargs["yticklabels"] = class_names

    if annotate_counts or annotate_percentages:
        if pct_matrix is None:
            row_sums = cm_counts.sum(axis=1, keepdims=True)
            pct_matrix = np.divide(
                cm_counts * 100.0,
                row_sums,
                out=np.zeros_like(cm_counts, dtype=float),
                where=row_sums != 0,
            )
        if annotate_counts and annotate_percentages:
            annot = np.empty_like(cm_counts, dtype=object)
            for i in range(cm_counts.shape[0]):
                for j in range(cm_counts.shape[1]):
                    pct_text = f"{pct_matrix[i, j]:.{annotation_decimals}f}"
                    if percentage_symbol:
                        pct_text = f"{pct_text}%"
                    annot[i, j] = f"{int(cm_counts[i, j])}\n{pct_text}"
            heatmap_kwargs["annot"] = annot
            heatmap_kwargs["fmt"] = ""
        elif annotate_percentages:
            annot = np.empty_like(cm_counts, dtype=object)
            for i in range(cm_counts.shape[0]):
                for j in range(cm_counts.shape[1]):
                    pct_text = f"{pct_matrix[i, j]:.{annotation_decimals}f}"
                    if percentage_symbol:
                        pct_text = f"{pct_text}%"
                    annot[i, j] = pct_text
            heatmap_kwargs["annot"] = annot
            heatmap_kwargs["fmt"] = ""
        else:
            heatmap_kwargs["annot"] = True
            heatmap_kwargs["fmt"] = "d"
    else:
        heatmap_kwargs["annot"] = False

    ax = sns.heatmap(cm_plot, **heatmap_kwargs)
    # Remove the outer axis box for cleaner publication-style figures.
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_title:
        if title_text is not None:
            plt.title(title_text, fontsize=20)
        else:
            title_suffix = " (Row-Normalized)" if normalize == "true" else ""
            plt.title(f"Confusion Matrix{title_suffix}", fontsize=20)
    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(rotation=90, fontsize=15)
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    return report
