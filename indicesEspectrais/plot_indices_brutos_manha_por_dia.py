#!/usr/bin/env python3
"""Índices WDI, NDVI e PRI calculados diretamente nas leituras brutas matinais.

Fonte: dataset/Unificada13052026_Limpa.csv. Não é aplicado pré-processamento
espectral (p. ex., suavização, normalização ou remoção de outliers): cada
índice é a fórmula direta sobre as reflectâncias de cada leitura.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_Limpa.csv"
GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]
CORES = {"IRRIG": "#2A9D8F", "NIRRIG": "#E76F51"}
T_CRITICO_95 = {31: 2.0395, 35: 2.0301}

INDICES = {
    "wdi": {
        "bandas": ["900", "970"],
        "rotulo": "WDI = R970 / R900",
        "titulo": "Índice de Déficit Hídrico (WDI)",
        "calcular": lambda d: d["970"] / d["900"],
    },
    "ndvi": {
        "bandas": ["670", "800"],
        "rotulo": "NDVI = (R800 − R670) / (R800 + R670)",
        "titulo": "Índice de Vegetação por Diferença Normalizada (NDVI)",
        "calcular": lambda d: (d["800"] - d["670"]) / (d["800"] + d["670"]),
    },
    "pri": {
        "bandas": ["531", "570"],
        "rotulo": "PRI = (R531 − R570) / (R531 + R570)",
        "titulo": "Índice de Reflectância Fotoquímica (PRI)",
        "calcular": lambda d: (d["531"] - d["570"]) / (d["531"] + d["570"]),
    },
}


def dias_ordenados(serie: pd.Series) -> pd.Categorical:
    dias = serie.str.extract(r"D(\d+)", expand=False).astype(int)
    return pd.Categorical(dias, categories=sorted(dias.unique()), ordered=True)


def resumo_diario(dados: pd.DataFrame, indice: str) -> pd.DataFrame:
    resumo = (dados.groupby(["genotipo", "condicao", "dia"], observed=True)[indice]
              .agg(media="mean", desvio_padrao="std", n="count").reset_index())
    erro = resumo["desvio_padrao"] / np.sqrt(resumo["n"])
    t_critico = (resumo["n"] - 1).map(T_CRITICO_95).fillna(1.96)
    margem = t_critico * erro
    resumo["ic95_inferior"] = resumo["media"] - margem
    resumo["ic95_superior"] = resumo["media"] + margem
    return resumo.sort_values(["genotipo", "condicao", "dia"])


def gerar(nome: str, config: dict, base: pd.DataFrame) -> None:
    bandas = config["bandas"]
    dados = base[["genotipo", "condicao", "data_coleta", *bandas]].copy()
    # Conversão de vírgula decimal é apenas leitura do CSV; não altera os espectros.
    for banda in bandas:
        dados[banda] = pd.to_numeric(
            dados[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        dados[nome] = config["calcular"](dados)
    dados = dados.replace([np.inf, -np.inf], np.nan).dropna(subset=[nome])
    dados["dia"] = dias_ordenados(dados["data_coleta"])
    resumo = resumo_diario(dados, nome)
    resumo.to_csv(ROOT / f"{nome}_bruto_manha_por_dia_resumo.csv", index=False,
                  float_format="%.6f")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(GENOTIPOS), figsize=(15, 4.8), sharey=True)
    dias = list(resumo["dia"].cat.categories)
    for ax, genotipo in zip(axes, GENOTIPOS):
        por_genotipo = resumo[resumo["genotipo"] == genotipo]
        for condicao in CONDICOES:
            grupo = por_genotipo[por_genotipo["condicao"] == condicao]
            x = grupo["dia"].astype(int)
            ax.plot(x, grupo["media"], marker="o", linewidth=2.4, markersize=6,
                    color=CORES[condicao], label=condicao)
            ax.fill_between(x, grupo["ic95_inferior"], grupo["ic95_superior"],
                            color=CORES[condicao], alpha=0.18)
        ax.set_title(genotipo, fontsize=15, fontweight="bold")
        ax.set_xticks(dias)
        ax.set_xticklabels([f"D{dia:02d}" for dia in dias])
        ax.set_xlabel("Dia de coleta", fontsize=12)
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(config["rotulo"], fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f"{config['titulo']} — leituras brutas da manhã", fontsize=15,
                 fontweight="bold")
    fig.text(0.5, 0.89, "Linhas = média das leituras brutas; faixas = IC de 95%",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.savefig(ROOT / f"{nome}_bruto_manha_por_dia_genotipos.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(ROOT / f"{nome}_bruto_manha_por_dia_genotipos.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = pd.read_csv(ENTRADA, sep=";")
    base = base.loc[(base["turno"].str.lower() == "manha")
                    & base["genotipo"].isin(GENOTIPOS)
                    & base["condicao"].isin(CONDICOES)].copy()
    for nome, config in INDICES.items():
        gerar(nome, config, base)
        print(f"{nome.upper()}: concluído")


if __name__ == "__main__":
    main()
