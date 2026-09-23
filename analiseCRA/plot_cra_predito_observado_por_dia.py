#!/usr/bin/env python3
"""Gráfico único de CRA observado versus predito em CV, colorido por data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "analiseCRA" / "resultados"
ARQUIVOS = {
    "24/02": ("24_02_predicoes_cv.csv", "CRA (24/02)"),
    "02/03": ("02_03_predicoes_cv.csv", "CRA (02/03)"),
    "03/03": ("03_03_predicoes_cv.csv", "CRA (03/03)"),
}
CORES = {"24/02": "#287a8e", "02/03": "#e8902f", "03/03": "#8c4f9d"}


def main():
    dados = []
    for dia, (arquivo, coluna_observado) in ARQUIVOS.items():
        tabela = pd.read_csv(RESULTADOS / arquivo, sep=";")
        dados.append(pd.DataFrame({
            "dia": dia,
            "cra_observado": tabela[coluna_observado],
            "cra_predito_cv": tabela["predito_cv"],
        }))
    dados = pd.concat(dados, ignore_index=True)

    limite_inferior = min(dados.cra_observado.min(), dados.cra_predito_cv.min())
    limite_superior = max(dados.cra_observado.max(), dados.cra_predito_cv.max())
    margem = (limite_superior - limite_inferior) * .05
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for dia in ARQUIVOS:
        parte = dados[dados.dia.eq(dia)]
        ax.scatter(parte.cra_observado, parte.cra_predito_cv, s=58, color=CORES[dia],
                   edgecolor="white", linewidth=.7, alpha=.9, label=dia)
    ax.plot([limite_inferior - margem, limite_superior + margem],
            [limite_inferior - margem, limite_superior + margem],
            "--", color="#333333", linewidth=1.2, label="1:1")
    ax.set(xlim=(limite_inferior - margem, limite_superior + margem),
           ylim=(limite_inferior - margem, limite_superior + margem),
           xlabel="CRA observado", ylabel="CRA predito (CV por bloco)",
           title="CRA: observado × predito por data")
    ax.legend(title="Data", frameon=True)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "cra_observado_x_predito_por_dia.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTADOS / "cra_observado_x_predito_por_dia.pdf", bbox_inches="tight")
    plt.close(fig)
    dados.to_csv(RESULTADOS / "cra_observado_x_predito_por_dia.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
