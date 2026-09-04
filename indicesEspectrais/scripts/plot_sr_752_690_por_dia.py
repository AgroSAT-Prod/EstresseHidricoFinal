#!/usr/bin/env python3
"""Calcula e representa a razão espectral SR 752/690 ao longo dos dias.

O índice é calculado em cada leitura como R752 / R690.  São usadas somente
as leituras do turno da manhã, disponíveis em todos os dias. Cada painel
corresponde a um genótipo; linhas representam as condições hídricas, com
faixas de intervalo de confiança de 95% entre leituras.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_Limpa.csv"
GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]
CORES_CONDICAO = {"IRRIG": "#2A9D8F", "NIRRIG": "#E76F51"}
# Todos os grupos desta base têm 32 ou 36 leituras. Valores de t bilateral
# para 95% de confiança (gl = n - 1), evitando dependência adicional de SciPy.
T_CRITICO_95 = {31: 2.0395, 35: 2.0301}


def ordenar_dias(serie: pd.Series) -> pd.Categorical:
    """Extrai o número de D02M e o ordena cronologicamente."""
    dias = serie.str.extract(r"D(\d+)", expand=False).astype(int)
    return pd.Categorical(dias, categories=sorted(dias.unique()), ordered=True)


def resumir_indice(dados: pd.DataFrame) -> pd.DataFrame:
    """Calcula média, desvio-padrão, n e IC95% por grupo."""
    resumo = (
        dados.groupby(["genotipo", "condicao", "dia"], observed=True)["sr_752_690"]
        .agg(media="mean", desvio_padrao="std", n="count")
        .reset_index()
    )
    erro = resumo["desvio_padrao"] / np.sqrt(resumo["n"])
    t_critico = (resumo["n"] - 1).map(T_CRITICO_95).fillna(1.96)
    margem = t_critico * erro
    resumo["ic95_inferior"] = resumo["media"] - margem
    resumo["ic95_superior"] = resumo["media"] + margem
    return resumo.sort_values(["genotipo", "condicao", "dia"])


def main() -> None:
    dados = pd.read_csv(ARQUIVO_ENTRADA, sep=";")
    dados = dados.loc[
        (dados["turno"].str.lower() == "manha")
        & dados["genotipo"].isin(GENOTIPOS)
        & dados["condicao"].isin(CONDICOES),
        ["genotipo", "condicao", "data_coleta", "690", "752"],
    ].copy()
    for banda in ("690", "752"):
        dados[banda] = pd.to_numeric(
            dados[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    dados = dados.dropna(subset=["690", "752"])
    dados = dados.loc[dados["690"] != 0].copy()
    dados["sr_752_690"] = dados["752"] / dados["690"]
    dados["dia"] = ordenar_dias(dados["data_coleta"])

    resumo = resumir_indice(dados)
    resumo.to_csv(ROOT / "resultados" / "sr_752_690_por_dia_resumo.csv", index=False, float_format="%.6f")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(GENOTIPOS), figsize=(15, 4.8), sharey=True)
    dias = list(resumo["dia"].cat.categories)
    for ax, genotipo in zip(axes, GENOTIPOS):
        para_genotipo = resumo[resumo["genotipo"] == genotipo]
        for condicao in CONDICOES:
            grupo = para_genotipo[para_genotipo["condicao"] == condicao]
            x = grupo["dia"].astype(int)
            ax.plot(x, grupo["media"], marker="o", linewidth=2.4, markersize=6,
                    color=CORES_CONDICAO[condicao], label=condicao)
            ax.fill_between(x, grupo["ic95_inferior"], grupo["ic95_superior"],
                            color=CORES_CONDICAO[condicao], alpha=0.18)
        ax.set_title(genotipo, fontsize=15, fontweight="bold")
        ax.set_xticks(dias)
        ax.set_xticklabels([f"D{dia:02d}" for dia in dias])
        ax.set_xlabel("Dia de coleta", fontsize=12)
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("SR = R752 / R690", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Razão Espectral SR 752/690 ao longo dos dias", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.89, "Turno da manhã; linhas = média das leituras e faixas = IC de 95%",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.savefig(ROOT / "resultados" / "sr_752_690_por_dia_genotipos.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "resultados" / "sr_752_690_por_dia_genotipos.pdf", bbox_inches="tight")
    print(f"Gráfico salvo em: {ROOT / "resultados" / "sr_752_690_por_dia_genotipos.png"}")


if __name__ == "__main__":
    main()
