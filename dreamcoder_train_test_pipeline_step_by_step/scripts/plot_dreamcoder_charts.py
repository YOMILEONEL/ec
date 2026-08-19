"""Regenerate the two DreamCoder comparison charts with the thesis's own
short metric names (Solve Rate/POS/PPS/PSS/PES). Values are loaded directly
from each configuration's step07_metrics_summary.json (abstract_* fields)
rather than hardcoded, so the charts can never drift out of sync with the
actual computed metrics the way a copy-pasted literal can. DreamCoder is a
single run per configuration; its seed parameter has no effect under this
thesis's configuration, see Evaluation.tex's Hyperparameters section.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ["Consolidation", "Recognition", "Combined"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]

RUN_DIRS = {
    "Consolidation": "train_T_2_train__test_T_2_test__ET_15_TT_15_it_3_MF_10_rec_false_nocons_false",
    "Recognition": "train_T_2_train__test_T_2_test__ET_15_TT_15_it_3_MF_10_rec_true_nocons_true",
    "Combined": "train_T_2_train__test_T_2_test__ET_15_TT_15_it_3_MF_10_rec_true_nocons_false",
}
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

IMAGES_DIR = Path("C:/Users/Lenovo PC/Downloads/Bachelorarbeit/repo/bachelorarbeit/arbeit/Latex_Template/Template 4/images")


def load_summary(config):
    p = OUTPUTS_DIR / RUN_DIRS[config] / "step07_calculate_metrics" / "step07_metrics_summary.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_series():
    summaries = {c: load_summary(c) for c in CONFIGS}
    solve_rate = [summaries[c]["test_accuracy"] for c in CONFIGS]
    metric_keys = {"POS": "operation_score", "PPS": "position_score", "PSS": "sequence_score", "PES": "edit_score"}
    all_task = {m: [summaries[c][f"abstract_{key}_mean"] for c in CONFIGS] for m, key in metric_keys.items()}
    solved_only = {m: [summaries[c][f"abstract_{key}_mean_solved_only"] for c in CONFIGS] for m, key in metric_keys.items()}
    return solve_rate, all_task, solved_only


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
    solve_rate, all_task, solved_only = load_series()
    main_series = {"Solve Rate": solve_rate, **all_task}
    bar_chart(
        ["Solve Rate", "POS", "PPS", "PSS", "PES"], main_series,
        "DreamCoder Metrics by Configuration (500 training tasks, 3 iterations)",
        0.45, IMAGES_DIR / "dreamcoder_metric_comparison.png",
    )
    bar_chart(
        ["POS", "PPS", "PSS", "PES"], solved_only,
        "DreamCoder Structural Metrics by Configuration (solved tasks only)",
        0.30, IMAGES_DIR / "dreamcoder_metric_comparison_solved.png",
    )


if __name__ == "__main__":
    main()
