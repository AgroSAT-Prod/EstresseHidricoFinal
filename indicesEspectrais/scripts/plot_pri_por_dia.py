#!/usr/bin/env python3
"""Calcula e representa o PRI temporal de BR16, CD202 e EMB48.

O PRI (Photochemical Reflectance Index) e calculado em cada leitura como:
    (R531 - R570) / (R531 + R570)

O grafico usa somente o turno da manha, que esta disponivel em todos os dias,
e mostra media e IC de 95% entre as leituras de cada genotipo e condicao.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_Limpa.csv"
GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]
CORES_CONDICAO = {"IRRIG": "#2A9D8F", "NIRRIG": "#E76F51"}


def ordenar_dias(serie: pd.Series) -> pd.Categorical:
    """Converte D02M em D02 e preserva a ordem cronologica."""
    dias = serie.str.extract(r"D(\d+)", expand=False).astype(int)
    ordem = sorted(dias.unique())
    return pd.Categorical(dias, categories=ordem, ordered=True)


def resumir_pri(dados: pd.DataFrame) -> pd.DataFrame:
    """Retorna media, desvio, tamanho amostral e IC95% por grupo."""
    resumo = (
        dados.groupby(["genotipo", "condicao", "dia"], observed=True)["pri"]
        .agg(media="mean", desvio_padrao="std", n="count")
        .reset_index()
    )
    erro = resumo["desvio_padrao"] / np.sqrt(resumo["n"])
    margem = t.ppf(0.975, resumo["n"] - 1) * erro
    resumo["ic95_inferior"] = resumo["media"] - margem
    resumo["ic95_superior"] = resumo["media"] + margem
    return resumo.sort_values(["genotipo", "condicao", "dia"])


def main() -> None:
    dados = pd.read_csv(ARQUIVO_ENTRADA, sep=";")
    dados = dados.loc[
        (dados["turno"].str.lower() == "manha")
        & dados["genotipo"].isin(GENOTIPOS)
        & dados["condicao"].isin(CONDICOES),
        ["genotipo", "condicao", "data_coleta", "531", "570"],
    ].copy()

    # A base usa virgula como separador decimal nas reflectancias.
    for banda in ("531", "570"):
        dados[banda] = pd.to_numeric(
            dados[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    dados = dados.dropna(subset=["531", "570"])
    dados["pri"] = (dados["531"] - dados["570"]) / (dados["531"] + dados["570"])
    dados["dia"] = ordenar_dias(dados["data_coleta"])

    resumo = resumir_pri(dados)
    resumo.to_csv(ROOT / "resultados" / "pri_por_dia_resumo.csv", index=False, float_format="%.6f")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(GENOTIPOS), figsize=(15, 4.8), sharey=True)
    dias = list(resumo["dia"].cat.categories)

    for ax, genotipo in zip(axes, GENOTIPOS):
        para_genotipo = resumo[resumo["genotipo"] == genotipo]
        for condicao in CONDICOES:
            grupo = para_genotipo[para_genotipo["condicao"] == condicao]
            ax.plot(grupo["dia"].astype(int), grupo["media"], marker="o", linewidth=2.4,
                    markersize=6, color=CORES_CONDICAO[condicao], label=condicao)
            ax.fill_between(grupo["dia"].astype(int), grupo["ic95_inferior"],
                            grupo["ic95_superior"], color=CORES_CONDICAO[condicao], alpha=0.18)
        ax.set_title(genotipo, fontsize=15, fontweight="bold")
        ax.set_xticks(dias)
        ax.set_xticklabels([f"D{dia:02d}" for dia in dias])
        ax.set_xlabel("Dia de coleta", fontsize=12)
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("PRI = (R531 − R570) / (R531 + R570)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Índice de Reflectância Fotoquímica (PRI) ao longo dos dias",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.89, "Turno da manhã; linhas = média das leituras e faixas = IC de 95%",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.savefig(ROOT / "resultados" / "pri_por_dia_genotipos.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "resultados" / "pri_por_dia_genotipos.pdf", bbox_inches="tight")
    print(f"Gráfico salvo em: {ROOT / "resultados" / "pri_por_dia_genotipos.png"}")


if __name__ == "__main__":
    main()
