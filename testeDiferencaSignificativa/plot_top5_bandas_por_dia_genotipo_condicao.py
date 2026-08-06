#!/usr/bin/env python3
"""Visualiza as cinco bandas mais discriminantes por dia, genótipo e condição.

Uso:
    python plot_top5_bandas_por_dia_genotipo_condicao.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / "dataset_gerado" / "top5_bandas_por_dia_genotipo_condicao.csv"
SAIDA = ROOT / "dataset_gerado" / "top5_bandas_por_dia_genotipo_condicao.png"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]

REGIOES = [
    ("VIS", 400, 700, "#e8f1fb"),
    ("Red edge", 700, 780, "#e5f6ed"),
    ("NIR", 780, 1350, "#fff3d6"),
    ("SWIR1", 1350, 1800, "#f9e7e9"),
    ("SWIR2", 1800, 2450, "#eee9f7"),
]


def main() -> None:
    if not ENTRADA.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {ENTRADA}")

    dados = pd.read_csv(ENTRADA, sep=";")
    fig, eixos = plt.subplots(
        len(GENOTIPOS), len(CONDICOES), figsize=(16, 11), sharex=True, sharey=True,
        layout="constrained",
    )
    cores = plt.get_cmap("viridis_r")(np.linspace(0.1, 0.9, 5))
    y_dia = {dia: i for i, dia in enumerate(DIAS)}

    for linha, genotipo in enumerate(GENOTIPOS):
        for coluna, condicao in enumerate(CONDICOES):
            ax = eixos[linha, coluna]
            for _, inicio, fim, cor in REGIOES:
                ax.axvspan(inicio, fim, color=cor, zorder=0)

            sub = dados[
                (dados["genotipo"] == genotipo) & (dados["condicao"] == condicao)
            ]
            for posicao in range(1, 6):
                pontos = sub[sub["posicao"] == posicao]
                y = pontos["dia"].map(y_dia).to_numpy() + (posicao - 3) * 0.11
                ax.scatter(
                    pontos["banda_nm"], y, s=45, color=cores[posicao - 1],
                    edgecolor="white", linewidth=0.8, zorder=3,
                )

            ax.set_title(f"{genotipo} — {condicao}", fontsize=12, fontweight="bold")
            ax.set_xlim(400, 2450)
            ax.set_ylim(len(DIAS) - 0.55, -0.55)
            ax.set_yticks(range(len(DIAS)), DIAS)
            ax.grid(axis="x", color="#777777", alpha=0.20, linewidth=0.6)

            if linha == 0:
                for nome, inicio, fim, _ in REGIOES:
                    ax.text((inicio + fim) / 2, -0.72, nome, ha="center", va="bottom",
                            fontsize=8, color="#444444")
            if coluna == 0:
                ax.set_ylabel("Dia")

    legenda = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cores[i],
               markeredgecolor="white", markersize=8, label=f"Posição {i + 1}")
        for i in range(5)
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.supxlabel("Comprimento de onda (nm)", y=0.045)
    fig.suptitle(
        "Top 5 bandas por dia, genótipo e condição\n"
        "Contrastes entre genótipos na mesma condição; p-value ≤ 0,05",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight")
    print(f"Gráfico salvo em {SAIDA}")


if __name__ == "__main__":
    main()
