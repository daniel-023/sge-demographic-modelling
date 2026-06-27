import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / 'sge_demographic_figures_cache'
(_CACHE_ROOT / 'matplotlib').mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / 'fontconfig').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_ROOT / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_ROOT / 'fontconfig'))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FONT_FAMILY = 'DejaVu Sans'
FIG_DPI = 200
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / 'figures'

PALETTE = {
    'line': '#b7c0ce',
    'text': '#1f2937',
    'muted': '#6b7280',
    'blue_edge': '#2563eb',
    'blue_fill': '#dbeafe',
    'green_edge': '#16a34a',
    'green_fill': '#dcfce7',
    'orange_edge': '#d97706',
    'orange_fill': '#fef3c7',
    'purple_edge': '#9333ea',
    'purple_fill': '#f3e8ff',
    'divider': '#d1d5db',
}


def rounded_box(ax, center, width, height, edge, face, text='', fs=14, weight='normal',
                roundness=0.008, lw=1.8, text_lines=None, text_color=None):
    x = center[0] - width / 2
    y = center[1] - height / 2
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={roundness}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(patch)
    tc = text_color or PALETTE['text']
    if text_lines:
        n = len(text_lines)
        line_h = 0.038
        top_y = center[1] + (n - 1) * line_h / 2
        for i, (txt, tfs, tw) in enumerate(text_lines):
            ax.text(center[0], top_y - i * line_h, txt,
                    ha='center', va='center', fontsize=tfs, color=tc,
                    weight=tw, fontname=FONT_FAMILY, zorder=4)
    elif text:
        ax.text(center[0], center[1], text,
                ha='center', va='center', fontsize=fs, color=tc,
                weight=weight, fontname=FONT_FAMILY, zorder=4)


def sliced_embedding(ax, center, width, height, edge, face, slices=4):
    x = center[0] - width / 2
    y = center[1] - height / 2
    outer = Rectangle((x, y), width, height, edgecolor=edge, facecolor=face, linewidth=1.8, zorder=4)
    ax.add_patch(outer)
    for index in range(1, slices):
        yline = y + height * index / slices
        ax.plot([x, x + width], [yline, yline], color=edge, linewidth=1.4, zorder=5)


def arrow(ax, start, end, color=None, lw=1.4, shrinkA=2, shrinkB=8):
    c = color or PALETTE['line']
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=14,
        linewidth=lw, color=c, shrinkA=shrinkA, shrinkB=shrinkB, zorder=2,
    ))


def line(ax, start, end, color=None, lw=1.4):
    c = color or PALETTE['line']
    ax.plot([start[0], end[0]], [start[1], end[1]], color=c, linewidth=lw, zorder=2)


def circle_node(ax, center, radius, edge, face):
    center_px = ax.transData.transform(center)
    y_edge_px = ax.transData.transform((center[0], center[1] + radius))
    radius_px = abs(y_edge_px[1] - center_px[1])
    radius_points = radius_px * 72.0 / ax.figure.dpi
    ax.plot(center[0], center[1], marker='o', markersize=2 * radius_points,
            markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.8,
            linestyle='None', zorder=4)


def draw_panel_a(ax):
    """Global-summary model (MLP) — left panel."""
    mid_x = 0.25

    # ── Title ──
    ax.text(mid_x, 0.92, 'A.  Global-summary model (MLP)',
            ha='center', va='center', fontsize=16, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY)

    # ── Row 1: Frame-level embeddings ──
    embed_y = 0.18
    embed_h = 0.14
    embed_w = 0.038
    x_positions = [0.10, 0.18, 0.26, 0.40]
    labels = [r'$h_1$', r'$h_2$', r'$h_3$', r'$h_T$']
    ellipsis_x = (x_positions[2] + x_positions[3]) / 2

    for x, lbl in zip(x_positions, labels):
        sliced_embedding(ax, (x, embed_y), embed_w, embed_h,
                         PALETTE['blue_edge'], PALETTE['blue_fill'], slices=4)

    # Labels below embeddings — shift down and use smaller font
    label_y = embed_y - embed_h / 2 - 0.035
    for x, lbl in zip(x_positions, labels):
        ax.text(x, label_y, lbl,
                ha='center', va='center', fontsize=12, color=PALETTE['text'],
                fontname=FONT_FAMILY)

    ax.text(ellipsis_x, embed_y, r'$\cdots$', ha='center', va='center',
            fontsize=22, color=PALETTE['muted'], fontname=FONT_FAMILY)

    ax.text(mid_x, 0.04, 'Captures global acoustic properties; discards temporal structure.',
            ha='center', va='center', fontsize=10.2, color=PALETTE['muted'],
            fontname=FONT_FAMILY)

    # ── Row 2: Mean pooling ──
    pool_y = 0.42
    pool_w, pool_h = 0.18, 0.065
    rounded_box(ax, (mid_x, pool_y), pool_w, pool_h,
                PALETTE['green_edge'], PALETTE['green_fill'],
                text='Mean Pooling', fs=12, weight='normal', roundness=0.006)
    ax.text(mid_x + pool_w / 2 + 0.07, pool_y, 'average all frames',
            ha='center', va='center', fontsize=9.5, color=PALETTE['muted'],
            fontname=FONT_FAMILY)

    # Arrows: embeddings → pooling (converging)
    for x in x_positions:
        arrow(ax, (x, embed_y + embed_h / 2), (mid_x, pool_y - pool_h / 2),
              shrinkA=2, shrinkB=6)

    # ── Row 3: Pooled embedding (single vector) ──
    pooled_y = 0.56
    pooled_w, pooled_h = 0.05, 0.10
    sliced_embedding(ax, (mid_x, pooled_y), pooled_w, pooled_h,
                     PALETTE['blue_edge'], PALETTE['blue_fill'], slices=4)
    ax.text(mid_x + pooled_w / 2 + 0.04, pooled_y, r'$\bar{h}$',
            ha='center', va='center', fontsize=15, color=PALETTE['text'],
            fontname=FONT_FAMILY)

    arrow(ax, (mid_x, pool_y + pool_h / 2), (mid_x, pooled_y - pooled_h / 2),
          shrinkA=4, shrinkB=4)

    # ── Row 4: MLP hidden layers ──
    mlp_y = 0.72
    mlp_w, mlp_h = 0.18, 0.065
    rounded_box(ax, (mid_x, mlp_y), mlp_w, mlp_h,
                PALETTE['green_edge'], PALETTE['green_fill'],
                text='Feedforward Layers', fs=12, weight='normal', roundness=0.006)

    arrow(ax, (mid_x, pooled_y + pooled_h / 2), (mid_x, mlp_y - mlp_h / 2),
          shrinkA=4, shrinkB=6)

    # ── Row 5: Output ──
    out_y = 0.86
    out_w, out_h = 0.18, 0.065
    rounded_box(ax, (mid_x, out_y), out_w, out_h,
                PALETTE['orange_edge'], PALETTE['orange_fill'],
                text='Prediction', fs=13, weight='bold', roundness=0.006)

    arrow(ax, (mid_x, mlp_y + mlp_h / 2), (mid_x, out_y - out_h / 2),
          shrinkA=4, shrinkB=6)


def draw_panel_b(ax):
    """Sequence-based model (LSTM) — right panel."""
    mid_x = 0.75

    # ── Title ──
    ax.text(mid_x, 0.92, 'B.  Sequence-based model (LSTM)',
            ha='center', va='center', fontsize=16, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY)

    # ── Row 1: Frame-level embeddings ──
    embed_y = 0.18
    embed_h = 0.14
    embed_w = 0.038
    x_positions = [0.60, 0.68, 0.76, 0.90]
    labels = [r'$h_1$', r'$h_2$', r'$h_3$', r'$h_T$']
    ellipsis_x = (x_positions[2] + x_positions[3]) / 2

    for x, lbl in zip(x_positions, labels):
        sliced_embedding(ax, (x, embed_y), embed_w, embed_h,
                         PALETTE['blue_edge'], PALETTE['blue_fill'], slices=4)

    label_y = embed_y - embed_h / 2 - 0.035
    for x, lbl in zip(x_positions, labels):
        ax.text(x, label_y, lbl,
                ha='center', va='center', fontsize=12, color=PALETTE['text'],
                fontname=FONT_FAMILY)

    ax.text(ellipsis_x, embed_y, r'$\cdots$', ha='center', va='center',
            fontsize=22, color=PALETTE['muted'], fontname=FONT_FAMILY)

    ax.text(mid_x, 0.04, 'Preserves time-varying rhythmic and intonational information.',
            ha='center', va='center', fontsize=10.2, color=PALETTE['muted'],
            fontname=FONT_FAMILY)

    # ── Row 2: LSTM cells ──
    lstm_y = 0.42
    lstm_w, lstm_h = 0.075, 0.058
    for x in x_positions:
        rounded_box(ax, (x, lstm_y), lstm_w, lstm_h,
                    PALETTE['green_edge'], PALETTE['green_fill'],
                    text='LSTM', fs=10.5, weight='normal', roundness=0.005)

    ax.text(ellipsis_x, lstm_y, r'$\cdots$', ha='center', va='center',
            fontsize=22, color=PALETTE['muted'], fontname=FONT_FAMILY)

    # Arrows: embeddings → LSTM cells (vertical, aligned)
    for x in x_positions:
        arrow(ax, (x, embed_y + embed_h / 2), (x, lstm_y - lstm_h / 2),
              shrinkA=2, shrinkB=6)

    # Temporal arrows between LSTM cells
    arrow(ax, (x_positions[0] + lstm_w / 2, lstm_y),
          (x_positions[1] - lstm_w / 2, lstm_y), shrinkA=4, shrinkB=6)
    arrow(ax, (x_positions[1] + lstm_w / 2, lstm_y),
          (x_positions[2] - lstm_w / 2, lstm_y), shrinkA=4, shrinkB=6)

    # ── Row 3: Sequence representation ──
    seq_y = 0.56
    seq_w, seq_h = 0.22, 0.065
    rounded_box(ax, (mid_x, seq_y), seq_w, seq_h,
                PALETTE['green_edge'], PALETTE['green_fill'],
                text='Aggregated Hidden States', fs=10.8, weight='normal', roundness=0.006)

    # Connector: all LSTM outputs → shared horizontal bar → sequence repr
    bar_y = 0.49
    for x in x_positions:
        line(ax, (x, lstm_y + lstm_h / 2), (x, bar_y), lw=1.1)
    line(ax, (x_positions[0], bar_y), (x_positions[-1], bar_y), lw=1.1)
    arrow(ax, (mid_x, bar_y), (mid_x, seq_y - seq_h / 2),
          shrinkA=0, shrinkB=6, lw=1.1)

    # ── Row 4: Classification layer ──
    cls_y = 0.72
    cls_w, cls_h = 0.22, 0.065
    rounded_box(ax, (mid_x, cls_y), cls_w, cls_h,
                PALETTE['green_edge'], PALETTE['green_fill'],
                text='Classification Layer', fs=12, weight='normal', roundness=0.006)

    arrow(ax, (mid_x, seq_y + seq_h / 2), (mid_x, cls_y - cls_h / 2),
          shrinkA=4, shrinkB=6)

    # ── Row 5: Output ──
    out_y = 0.86
    out_w, out_h = 0.18, 0.065
    rounded_box(ax, (mid_x, out_y), out_w, out_h,
                PALETTE['orange_edge'], PALETTE['orange_fill'],
                text='Prediction', fs=13, weight='bold', roundness=0.006)

    arrow(ax, (mid_x, cls_y + cls_h / 2), (mid_x, out_y - out_h / 2),
          shrinkA=4, shrinkB=6)


def main():
    plt.rcParams['font.family'] = FONT_FAMILY

    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=FIG_DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.50, 0.975, 'Comparison of the two modelling approaches',
            ha='center', va='center', fontsize=20, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY)

    # ── Divider ──
    ax.plot([0.50, 0.50], [0.04, 0.92], color=PALETTE['divider'],
            linewidth=1.2, linestyle='--', zorder=1)

    # ── Draw both panels ──
    draw_panel_a(ax)
    draw_panel_b(ax)

    plt.tight_layout(pad=0.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'pooled_vs_sequence.png'
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
