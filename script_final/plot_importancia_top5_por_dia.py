"""Plot the five most important RF bands for each day."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parent / "outputs" / "classificacao"


def plot_importance():
    data = pd.read_csv(OUTPUT / "top5_rf_por_dia.csv", sep=";")
    data = data.sort_values(["dia", "posicao"]).copy()
    days = sorted(data["dia"].unique())
    data["row"] = data["dia"].map({day: i for i, day in enumerate(days)}) * 6 + data["posicao"]

    low, high = data["importancia_gini"].min(), data["importancia_gini"].max()
    data["gini_normalized"] = (data["importancia_gini"] - low) / (high - low) if high > low else 0.0

    fig, ax = plt.subplots(figsize=(13, max(5, 1.65 * len(days) + 1.5)), constrained_layout=True)
    points = ax.scatter(
        data["banda_nm"], data["row"], c=data["gini_normalized"],
        cmap="viridis", vmin=0, vmax=1, s=105, edgecolors="#333333",
        linewidths=0.5, zorder=3,
    )
    for row in data.itertuples():
        ax.annotate(
            f"{int(row.banda_nm)} nm",
            (row.banda_nm, row.row), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=11,
        )
    for index in range(1, len(days)):
        ax.axhline(index * 6 + 0.5, color="#dddddd", linewidth=0.8)
    ax.set_yticks([i * 6 + 3 for i in range(len(days))], labels=days)
    ax.set_ylim(len(days) * 6 + 0.5, 0)
    ax.set_xlim(380, 2470)
    ax.set_xlabel("Wavelength (nm)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Day (bands ranked by importance)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", alpha=0.25)
    colorbar = fig.colorbar(points, ax=ax, ticks=np.linspace(0, 1, 6))
    colorbar.set_label("Normalized Gini importance (0–1)", fontsize=13, fontweight="bold")
    colorbar.ax.tick_params(labelsize=12)
    fig.savefig(OUTPUT / "importancia_top5_por_dia.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_importance()
