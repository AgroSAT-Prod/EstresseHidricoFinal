#!/usr/bin/env python3
"""Gráfico de predição CV do PLSR único de Clorofila Total."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent / "resultados" / "todos_dias"
ORDEM = ["23/02", "24/02", "25/02", "26/02", "27/02", "02/03", "03/03"]


def main():
    pred = pd.read_csv(OUT / "predicoes_cv_bloco.csv", sep=";")
    met = pd.read_csv(OUT / "resumo.csv", sep=";").iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    cores = plt.get_cmap("tab10").colors
    for cor, data in zip(cores, ORDEM):
        d = pred[pred.data.eq(data)]
        ax.scatter(d.chl_total, d.predito_cv, s=46, color=cor, alpha=.8,
                   edgecolor="white", linewidth=.5, label=data)
    minimo = min(pred.chl_total.min(), pred.predito_cv.min()) - .7
    maximo = max(pred.chl_total.max(), pred.predito_cv.max()) + .7
    ax.plot([minimo, maximo], [minimo, maximo], "--", color="#b43c2f", lw=1.6, label="1:1")
    ax.set(xlim=(minimo, maximo), ylim=(minimo, maximo), aspect="equal",
           xlabel="CHL Total observado", ylabel="CHL Total predito (CV por bloco)",
           title="PLSR — Clorofila Total, sete coletas combinadas")
    ax.grid(alpha=.18)
    ax.text(.04, .95, f"R² CV = {met.r2_cv_bloco:.3f}\nRMSE CV = {met.rmse_cv_bloco:.3f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#a0a0a0"})
    ax.legend(title="Data", loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "predicao_cv_todos_dias.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "predicao_cv_todos_dias.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
