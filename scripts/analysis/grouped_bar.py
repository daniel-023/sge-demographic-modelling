import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data: averaged across listener ethnicities per speaker/cohort
data = {
    "Chinese": {
        "19–29": (80.05 + 69.17 + 71.43) / 3,
        "30–49": (82.22 + 84.48 + 74.19) / 3,
        "50–65": (82.18 + 87.78 + 64.58) / 3,
    },
    "Malay": {
        "19–29": (39.39 + 59.17 + 40.00) / 3,
        "30–49": (57.78 + 66.09 + 56.45) / 3,
        "50–65": (60.34 + 68.33 + 59.90) / 3,
    },
    "Indian": {
        "19–29": (33.08 + 32.08 + 42.38) / 3,
        "30–49": (41.11 + 49.43 + 57.53) / 3,
        "50–65": (64.37 + 62.78 + 66.67) / 3,
    },
}

cohorts = ["19–29", "30–49", "50–65"]
speaker_ethnicities = ["Chinese", "Malay", "Indian"]
colors = ["#89C2D9", "#F4A896", "#CDB4DB"]  # soft teal, peach, lavender

x = np.arange(len(speaker_ethnicities))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))

for i, cohort in enumerate(cohorts):
    values = [data[eth][cohort] for eth in speaker_ethnicities]
    bars = ax.bar(x + i * width, values, width, label=cohort, color=colors[i], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8, color="#333333"
        )

ax.set_xlabel("Speaker Ethnicity", fontsize=12, labelpad=10)
ax.set_ylabel("Average Identification Accuracy (%)", fontsize=12, labelpad=10)
ax.set_xticks(x + width)
ax.set_xticklabels(speaker_ethnicities, fontsize=11)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title="Listener Age Group", fontsize=10, title_fontsize=10, framealpha=0.8)

plt.tight_layout()
out_path = Path(__file__).resolve().parents[2] / "figures" / "grouped_bar_chart.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Chart saved to {out_path}")
