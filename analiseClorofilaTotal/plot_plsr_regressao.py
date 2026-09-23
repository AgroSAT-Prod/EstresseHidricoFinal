#!/usr/bin/env python3
"""Painel observado × predito (CV por bloco) dos PLSR de Clorofila Total."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados"
DATAS = {
    "23_02": "23/02", "24_02": "24/02", "25_02": "25/02",
    "26_02": "26/02", "27_02": "27/02", "02_03": "02/03",
    "03_03": "03/03",
}


def main():
    resumo = pd.read_csv(OUT / "resumo_plsr_clorofila_total.csv", sep=";")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
    axes = axes.ravel()
    for ax, (codigo, data) in zip(axes, DATAS.items()):
        pred = pd.read_csv(OUT / f"{codigo}_predicoes_cv.csv", sep=";")
        alvo = next(c for c in pred.columns if c.startswith("CHL TOTAL"))
        r = resumo.loc[resumo["data_yield"].eq(data)].iloc[0]
        x, y = pred[alvo], pred["predito_cv"]
        margem = (max(x.max(), y.max()) - min(x.min(), y.min())) * .08
        minimo, maximo = min(x.min(), y.min()) - margem, max(x.max(), y.max()) + margem
        ax.scatter(x, y, s=54, color="#147d91", edgecolor="white", linewidth=.7, alpha=.9)
        ax.plot([minimo, maximo], [minimo, maximo], "--", color="#bb4430", lw=1.4, label="1:1")
        ax.set(xlim=(minimo, maximo), ylim=(minimo, maximo), aspect="equal",
               title=f"{r['data_yield']}  |  {r['coleta_espectral']}",
               xlabel="CHL Total observado", ylabel="CHL Total predito (CV)")
        ax.grid(alpha=.18)
        ax.text(.04, .95, f"R² CV = {r['r2_cv_bloco']:.3f}\nRMSE = {r['rmse_cv_bloco']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=.3", "facecolor": "white", "edgecolor": "#aaaaaa", "alpha": .9})
    axes[-1].axis("off")
    fig.suptitle("PLSR — Clorofila Total: observado × predito em validação por bloco", fontsize=16, fontweight="bold", y=.98)
    fig.text(.5, .02, "Cada ponto é uma unidade bloco × genótipo × condição; predições obtidas fora da dobra correspondente.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(OUT / "plsr_regressao_observado_predito_cv.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "plsr_regressao_observado_predito_cv.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
