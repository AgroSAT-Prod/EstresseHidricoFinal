#!/usr/bin/env python3
"""Clorofila total observada versus predita em CV, por data e com R² global."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "analiseClorofilaTotal" / "resultados"
ARQUIVOS = {
    "23/02": ("23_02_predicoes_cv.csv", "CHL TOTAL (23/02)"),
    "24/02": ("24_02_predicoes_cv.csv", "CHL TOTAL (24/02)"),
    "25/02": ("25_02_predicoes_cv.csv", "CHL TOTAL (25.02)"),
    "26/02": ("26_02_predicoes_cv.csv", "CHL TOTAL (26/02)"),
    "27/02": ("27_02_predicoes_cv.csv", "CHL TOTAL (27/02)"),
    "02/03": ("02_03_predicoes_cv.csv", "CHL TOTAL (02/03)"),
    "03/03": ("03_03_predicoes_cv.csv", "CHL TOTAL (03/03)"),
}
CORES = {
    "23/02": "#287a8e", "24/02": "#e8902f", "25/02": "#8c4f9d",
    "26/02": "#4c9a4d", "27/02": "#c8513d", "02/03": "#6d8fc1",
    "03/03": "#9c8c42",
}


def main():
    partes = []
    for dia, (arquivo, coluna_observada) in ARQUIVOS.items():
        tabela = pd.read_csv(RESULTADOS / arquivo, sep=";")
        partes.append(pd.DataFrame({
            "dia": dia,
            "clorofila_total_observada": tabela[coluna_observada],
            "clorofila_total_predita_cv": tabela["predito_cv"],
        }))
    dados = pd.concat(partes, ignore_index=True).dropna()
    r2_global = r2_score(dados.clorofila_total_observada, dados.clorofila_total_predita_cv)

    minimo = min(dados.clorofila_total_observada.min(), dados.clorofila_total_predita_cv.min())
    maximo = max(dados.clorofila_total_observada.max(), dados.clorofila_total_predita_cv.max())
    margem = (maximo - minimo) * .05
    fig, ax = plt.subplots(figsize=(8.3, 6.8))
    for dia in ARQUIVOS:
        parte = dados[dados.dia.eq(dia)]
        ax.scatter(parte.clorofila_total_observada, parte.clorofila_total_predita_cv,
                   s=56, color=CORES[dia], edgecolor="white", linewidth=.7,
                   alpha=.9, label=dia)
    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem],
            "--", color="#333333", linewidth=1.2, label="1:1")
    ax.text(.03, .97, f"R² global (CV por bloco) = {r2_global:.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=11,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#666", "alpha": .92})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel="Clorofila total observada", ylabel="Clorofila total predita (CV por bloco)",
           title="Clorofila total: observado × predito por data")
    ax.legend(title="Data", frameon=True, ncol=2, loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "clorofila_total_observado_x_predito_por_dia.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTADOS / "clorofila_total_observado_x_predito_por_dia.pdf", bbox_inches="tight")
    plt.close(fig)
    dados.to_csv(RESULTADOS / "clorofila_total_observado_x_predito_por_dia.csv", sep=";", index=False)
    pd.DataFrame([{"n_total": len(dados), "r2_global_cv_bloco": r2_global}]).to_csv(
        RESULTADOS / "clorofila_total_observado_x_predito_metricas.csv", sep=";", index=False
    )


if __name__ == "__main__":
    main()
