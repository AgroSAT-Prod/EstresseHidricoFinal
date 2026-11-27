"""Regenerate the FWHM classification figures from saved result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estilo_graficos import (
    AXIS_TITLE_SIZE,
    BAR_VALUE_SIZE,
    COMPACT_AXIS_TITLE_SIZE,
    LEGEND_SIZE,
    MATRIX_VALUE_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


OUTPUT = Path(__file__).resolve().parent / "outputs" / "classificacao" / "fwhm"
CLASSES = ["IRRIG", "NIRRIG"]


def plot_fwhm():
    counts = pd.read_csv(OUTPUT / "matrizes_fwhm_por_dia.csv", sep=";")
    metrics = pd.read_csv(OUTPUT / "metricas_xgboost_por_dia.csv", sep=";")
    days = sorted(counts["dia"].unique())
    matrices = {
        day: counts.loc[counts["dia"] == day]
        .set_index("condicao_real")
        .loc[CLASSES, ["previsto_IRRIG", "previsto_NIRRIG"]]
        .to_numpy()
        for day in days
    }

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    for index, (ax, day) in enumerate(zip(axes.flat, days)):
        matrix = matrices[day]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_title(f"({chr(97 + index)}) {day}", fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_xlabel("Predicted", fontsize=COMPACT_AXIS_TITLE_SIZE, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=COMPACT_AXIS_TITLE_SIZE, fontweight="bold")
        for row in range(2):
            total = matrix[row].sum()
            for col in range(2):
                n = int(matrix[row, col])
                percentage = 100 * n / total if total else 0
                ax.text(col, row, f"{n}\n({percentage:.1f}%)", ha="center", va="center",
                        fontsize=MATRIX_VALUE_SIZE, color="white" if n > maximum / 2 else "#1d2a35")
    for ax in list(axes.flat)[len(days):]:
        ax.axis("off")
    fig.savefig(OUTPUT / "matrizes_fwhm_por_dia.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    series = (
        ("acuracia", "Accuracy", "#2a78d6"),
        ("acuracia_balanceada", "Balanced accuracy", "#1baf7a"),
        ("f1_nirrig", "F1 NIRRIG", "#eb6834"),
        ("auc_roc", "ROC AUC", "#7554a3"),
    )
    positions = np.arange(len(metrics))
    width = 0.20
    for index, (column, label, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(positions + offset, metrics[column], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=BAR_VALUE_SIZE, rotation=90)
    ax.set_ylim(0, 1.20)
    ax.set_xticks(positions, [f"({chr(97 + index)})\n{day}" for index, day in enumerate(metrics["dia"])])
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Day", fontsize=AXIS_TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Metric value", fontsize=AXIS_TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="upper center", fontsize=LEGEND_SIZE)
    fig.savefig(OUTPUT / "metricas_fwhm_por_dia.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_fwhm()
