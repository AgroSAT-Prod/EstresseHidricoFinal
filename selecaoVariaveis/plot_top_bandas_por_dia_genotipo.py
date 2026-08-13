#!/usr/bin/env python3
"""Plota as Top 5 por dia, genotipo e correlacao maxima aceita."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ENTRADA_PADRAO = (
    ROOT / "dataset_gerado" / "experimento_spearman_por_dia"
    / "top5_bandas.csv"
)
SAIDA_PADRAO = ROOT / "top_bandas_por_dia_genotipo.png"

CORES = {"BR16": "#2878B5", "CD202": "#E69F00", "EMB48": "#C63D4B"}
DESLOCAMENTOS = {"BR16": -0.18, "CD202": 0.0, "EMB48": 0.18}


def plotar(dados: pd.DataFrame, saida: Path) -> None:
    dias = sorted(dados["dia"].dropna().unique())
    genotipos = [g for g in CORES if g in set(dados["genotipo"])]
    correlacoes_maximas = sorted(
        dados["correlacao_maxima_aceita_top"].unique()
    )
    y_dia = {dia: i for i, dia in enumerate(dias)}

    fig, axes = plt.subplots(
        1, len(correlacoes_maximas),
        figsize=(6 * len(correlacoes_maximas), 6),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, correlacao_maxima in zip(axes, correlacoes_maximas):
        sub_limiar = dados[
            np.isclose(
                dados["correlacao_maxima_aceita_top"], correlacao_maxima
            )
        ]
        for genotipo in genotipos:
            sub = sub_limiar[sub_limiar["genotipo"] == genotipo]
            for dia, sub_dia in sub.groupby("dia", sort=False):
                bandas = np.sort(sub_dia["banda_nm"].to_numpy())
                y_linha = y_dia[dia] + DESLOCAMENTOS[genotipo]
                ax.plot(
                    bandas, np.full(len(bandas), y_linha),
                    color=CORES[genotipo], linewidth=7.0, alpha=0.20,
                    solid_capstyle="round", zorder=1,
                )
            y = sub["dia"].map(y_dia).to_numpy() + DESLOCAMENTOS[genotipo]
            ax.scatter(
                sub["banda_nm"], y, s=38, alpha=0.82,
                color=CORES[genotipo], edgecolor="white", linewidth=0.4,
                label=genotipo, zorder=2,
            )

        ax.set_title(
            f"Limiar {correlacao_maxima:.2f}: grupos |ρ| >; Top |ρ| <",
            weight="bold",
        )
        ax.set_xlabel("Comprimento de onda (nm)")
        ax.set_xlim(380, 2470)
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(range(len(dias)), dias)
    axes[0].set_ylabel("Dia de coleta")
    axes[-1].legend(title="Genotipo", frameon=True, loc="best")
    fig.suptitle(
        "Top 5 bandas espectrais por dia e genotipo — turno da manha",
        fontsize=14, weight="bold",
    )

    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=ENTRADA_PADRAO)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()

    if not args.entrada.exists():
        raise SystemExit(f"Arquivo nao encontrado: {args.entrada}")
    dados = pd.read_csv(args.entrada, sep=";")
    if dados.empty:
        raise SystemExit("O arquivo de Top 5 esta vazio.")

    plotar(dados, args.saida)
    print(f"Grafico salvo em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
