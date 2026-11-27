"""Regenerate daily confusion matrices with the shared chart style."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from estilo_graficos import (
    COMPACT_AXIS_TITLE_SIZE,
    MATRIX_VALUE_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


OUTPUT = Path(__file__).resolve().parent / "outputs" / "classificacao"
CLASSES = ["IRRIG", "NIRRIG"]


def plot_confusion_matrices():
    counts = pd.read_csv(OUTPUT / "matrizes_confusao_por_dia.csv", sep=";")
    days = sorted(counts["dia"].unique())
    matrices = {
        day: counts.loc[counts["dia"] == day]
        .set_index("condicao_real")
        .loc[CLASSES, ["previsto_IRRIG", "previsto_NIRRIG"]]
        .to_numpy()
        for day in days
    }
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
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
                        fontsize=MATRIX_VALUE_SIZE,
                        color="white" if n > maximum / 2 else "#1d2a35")
    for ax in list(axes.flat)[len(days):]:
        ax.axis("off")
    fig.savefig(OUTPUT / "matrizes_confusao_por_dia.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_confusion_matrices()
