"""Regenerate the two DreamCoder comparison charts with the thesis's own
short metric names (BSS/POS/PPS/PSS/PES) instead of the old pipeline names
(Accuracy, ProgramOperationScore, ...), matching the terminology fix already
applied to the DeepCoder charts. The underlying numbers are unchanged and
match Tables 6.1-6.5 in Results.tex exactly (DreamCoder is a single run per
configuration; its seed parameter has no effect under this thesis's
configuration, see Evaluation.tex's Hyperparameters section).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ["Consolidation", "Recognition", "Combined"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]

# All-task values (solve rate as a fraction, then POS/PPS/PSS/PES all-task means).
SOLVE_RATE = [0.253, 0.313, 0.343]
ALL_TASK = {
    "POS": [0.048, 0.063, 0.055],
    "PPS": [0.020, 0.017, 0.031],
    "PSS": [0.040, 0.052, 0.046],
    "PES": [0.035, 0.046, 0.049],
}
SOLVED_ONLY = {
    "POS": [0.191, 0.202, 0.160],
    "PPS": [0.080, 0.053, 0.092],
    "PSS": [0.158, 0.165, 0.133],
    "PES": [0.138, 0.148, 0.143],
}

IMAGES_DIR = Path("C:/Users/Lenovo PC/Downloads/Bachelorarbeit/repo/bachelorarbeit/arbeit/Latex_Template/Template 4/images")


def bar_chart(labels, series_by_config, title, ylim, outpath):
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, config in enumerate(CONFIGS):
        vals = [series_by_config[label][i] for label in labels]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=config, color=COLORS[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + ylim * 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_ylim(0, ylim)
    ax.set_ylabel("Mean")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", outpath)


def main():
    main_series = {"BSS": SOLVE_RATE, **ALL_TASK}
    bar_chart(
        ["BSS", "POS", "PPS", "PSS", "PES"], main_series,
        "DreamCoder Metrics by Configuration (500 training tasks, 3 iterations)",
        0.45, IMAGES_DIR / "dreamcoder_metric_comparison.png",
    )
    bar_chart(
        ["POS", "PPS", "PSS", "PES"], SOLVED_ONLY,
        "DreamCoder Structural Metrics by Configuration (solved tasks only)",
        0.30, IMAGES_DIR / "dreamcoder_metric_comparison_solved.png",
    )


if __name__ == "__main__":
    main()
