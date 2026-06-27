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
    'line': '#4b5563',
    'line_light': '#b7c0ce',
    'text': '#1f2937',
    'muted': '#6b7280',
    'blue_edge': '#6b7fad',
    'blue_fill': '#e8edf5',
    'green_edge': '#4a8c6a',
    'green_fill': '#dff0e6',
    'yellow_edge': '#c9a227',
    'yellow_fill': '#fef9e7',
    'purple_edge': '#8b6aad',
    'purple_fill': '#f0e8f7',
    'purple_outer': '#a88cc4',
}


def rounded_box(ax, center, width, height, edge, face, text='', fs=14, weight='normal',
                roundness=0.012, text_color=None, lw=1.8, text_lines=None):
    x = center[0] - width / 2
    y = center[1] - height / 2
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.008,rounding_size={roundness}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(patch)
    tc = text_color or PALETTE['text']
    if text_lines:
        n = len(text_lines)
        line_h = 0.032
        top_y = center[1] + (n - 1) * line_h / 2
        for i, (txt, tfs, tw) in enumerate(text_lines):
            ax.text(center[0], top_y - i * line_h, txt,
                    ha='center', va='center', fontsize=tfs, color=tc,
                    weight=tw, fontname=FONT_FAMILY, zorder=4)
    elif text:
        ax.text(center[0], center[1], text,
                ha='center', va='center', fontsize=fs, color=tc,
                weight=weight, fontname=FONT_FAMILY, zorder=4)


def pill_label(ax, center, text, edge, face, fs=13, width=0.14, height=0.032):
    rounded_box(ax, center, width, height, edge, face, text, fs=fs,
                weight='normal', roundness=0.015, lw=1.2)


def arrow(ax, start, end, color=None, lw=1.6, shrinkA=2, shrinkB=8, mut_scale=16):
    c = color or PALETTE['line']
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=mut_scale,
        linewidth=lw, color=c, shrinkA=shrinkA, shrinkB=shrinkB, zorder=2,
    ))


def line(ax, start, end, color=None, lw=1.4):
    c = color or PALETTE['line']
    ax.plot([start[0], end[0]], [start[1], end[1]], color=c, linewidth=lw, zorder=2)


def waveform(ax, center, width, height):
    """Draw a stylised waveform."""
    n = 180
    x = np.linspace(center[0] - width / 2 + 0.01, center[0] + width / 2 - 0.01, n)
    envelope = np.exp(-0.5 * ((x - center[0]) / (width * 0.22)) ** 2)
    np.random.seed(42)
    noise = np.random.randn(n) * 0.4
    y = center[1] + envelope * noise * height * 0.45
    ax.plot(x, y, color='#1f2937', linewidth=0.8, zorder=5)


def main():
    plt.rcParams['font.family'] = FONT_FAMILY

    fig, ax = plt.subplots(figsize=(14.5, 7.5), dpi=FIG_DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── Title ──
    ax.text(0.50, 0.95, 'Overview of the demographic prediction pipeline',
            ha='center', va='center', fontsize=22, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY)

    # ── Column header pills ──
    pill_y = 0.86
    cols = [
        (0.115, '1. Input', PALETTE['blue_edge'], PALETTE['blue_fill']),
        (0.355, '2. Speech encoder', PALETTE['green_edge'], PALETTE['green_fill']),
        (0.615, '3. Classifier heads', PALETTE['yellow_edge'], PALETTE['yellow_fill']),
        (0.875, '4. Outputs', PALETTE['purple_edge'], PALETTE['purple_fill']),
    ]
    for cx, txt, edge, face in cols:
        pill_label(ax, (cx, pill_y), txt, edge, face, fs=12.5, width=0.175, height=0.035)

    # ── Column 1: Input ──
    input_center = (0.115, 0.55)
    input_w, input_h = 0.18, 0.42
    rounded_box(ax, input_center, input_w, input_h, PALETTE['blue_edge'], PALETTE['blue_fill'],
                roundness=0.012, lw=1.4)
    ax.text(input_center[0], input_center[1] + 0.13, 'National Speech Corpus',
            ha='center', va='center', fontsize=11.5, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY, zorder=5)
    ax.text(input_center[0], input_center[1] + 0.09, 'speech recordings',
            ha='center', va='center', fontsize=12.5, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY, zorder=5)
    waveform(ax, (input_center[0], input_center[1] - 0.03), input_w * 0.85, input_h * 0.4)

    # ── Column 2: WavLM ──
    wavlm_center = (0.355, 0.55)
    wavlm_w, wavlm_h = 0.175, 0.20
    rounded_box(ax, wavlm_center, wavlm_w, wavlm_h, PALETTE['green_edge'], PALETTE['green_fill'],
                roundness=0.012, lw=1.6,
                text_lines=[
                    ('WavLM', 16, 'bold'),
                    ('extracts frame-level', 11, 'normal'),
                    ('representations', 11, 'normal'),
                ])

    # Arrow: input → WavLM
    arrow(ax, (input_center[0] + input_w / 2, input_center[1]),
          (wavlm_center[0] - wavlm_w / 2, wavlm_center[1]),
          shrinkA=4, shrinkB=6)

    # ── Column 3: Classifier heads ──
    # Fork point
    fork_x = wavlm_center[0] + wavlm_w / 2 + 0.035
    mlp_y = 0.67
    lstm_y = 0.40

    # Horizontal line from WavLM to fork
    line(ax, (wavlm_center[0] + wavlm_w / 2, wavlm_center[1]),
         (fork_x, wavlm_center[1]))
    # Vertical fork
    line(ax, (fork_x, mlp_y), (fork_x, lstm_y))

    # MLP box
    mlp_center = (0.615, mlp_y)
    mlp_w, mlp_h = 0.24, 0.18
    rounded_box(ax, mlp_center, mlp_w, mlp_h, PALETTE['yellow_edge'], PALETTE['yellow_fill'],
                roundness=0.012, lw=1.6,
                text_lines=[
                    ('Global-summary head (MLP)', 11.5, 'bold'),
                    ('', 4, 'normal'),
                    ('Averages across the utterance', 10.5, 'normal'),
                    ('before classification.', 10.5, 'normal'),
                ])

    # LSTM box
    lstm_center = (0.615, lstm_y)
    lstm_w, lstm_h = 0.24, 0.18
    rounded_box(ax, lstm_center, lstm_w, lstm_h, PALETTE['yellow_edge'], PALETTE['yellow_fill'],
                roundness=0.012, lw=1.6,
                text_lines=[
                    ('Sequence-based head (LSTM)', 11.5, 'bold'),
                    ('', 4, 'normal'),
                    ('Preserves temporal order', 10.5, 'normal'),
                    ('before classification.', 10.5, 'normal'),
                ])

    # Arrows: fork → classifier boxes
    arrow(ax, (fork_x, mlp_y), (mlp_center[0] - mlp_w / 2, mlp_y), shrinkA=0, shrinkB=6)
    arrow(ax, (fork_x, lstm_y), (lstm_center[0] - lstm_w / 2, lstm_y), shrinkA=0, shrinkB=6)

    # ── Column 4: Outputs ──
    output_x = 0.875
    output_outer_center = (output_x, 0.535)
    output_outer_w, output_outer_h = 0.17, 0.48
    # Outer container
    outer_x = output_outer_center[0] - output_outer_w / 2
    outer_y = output_outer_center[1] - output_outer_h / 2
    outer_patch = FancyBboxPatch(
        (outer_x, outer_y), output_outer_w, output_outer_h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.4, edgecolor=PALETTE['purple_outer'], facecolor='#faf7fd', zorder=2,
    )
    ax.add_patch(outer_patch)

    ax.text(output_x, 0.73, 'Demographic attributes',
            ha='center', va='center', fontsize=12, weight='bold',
            color=PALETTE['text'], fontname=FONT_FAMILY, zorder=4)

    attr_labels = ['Age', 'Gender', 'Ethnicity']
    attr_ys = [0.64, 0.535, 0.43]
    for label, ay in zip(attr_labels, attr_ys):
        rounded_box(ax, (output_x, ay), 0.12, 0.07, PALETTE['purple_edge'], PALETTE['purple_fill'],
                    text=label, fs=14, weight='bold', roundness=0.012, lw=1.2)

    # Arrows: classifier boxes → output container
    arrow(ax, (mlp_center[0] + mlp_w / 2, mlp_y),
          (outer_x, output_outer_center[1] + 0.10),
          shrinkA=4, shrinkB=6)
    arrow(ax, (lstm_center[0] + lstm_w / 2, lstm_y),
          (outer_x, output_outer_center[1] - 0.10),
          shrinkA=4, shrinkB=6)

    # ── Caption ──
    ax.text(0.50, 0.10,
            'WavLM frame-level representations feed two classifier heads: a pooled MLP and a temporal LSTM.',
            ha='center', va='center', fontsize=12.5, style='italic',
            color=PALETTE['muted'], fontname=FONT_FAMILY)

    plt.tight_layout(pad=0.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'pipeline_overview.png'
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
