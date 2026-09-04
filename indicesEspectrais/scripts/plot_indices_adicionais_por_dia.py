#!/usr/bin/env python3
"""Calcula índices espectrais temporais, separados por genótipo.

São consideradas apenas leituras matinais. Os índices e bandas (nm) são:
NDRE=(R790-R720)/(R790+R720); REIP=700+40*(((R670+R780)/2-R700)/(R740-R700));
DD=(R749-R720)-(R701-R672); GNDVI=(R800-R550)/(R800+R550);
NDMI=(R860-R1240)/(R860+R1240); MSI=R1600/R820; WDI=R970/R900.
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
CORES = {"IRRIG": "#2A9D8F", "NIRRIG": "#E76F51"}
T_CRITICO_95 = {31: 2.0395, 35: 2.0301}

INDICES = {
    "ndre": {
        "bandas": ["720", "790"],
        "rotulo": "NDRE = (R790 − R720) / (R790 + R720)",
        "titulo": "Índice de Vegetação por Diferença Normalizada no Red-Edge (NDRE)",
        "calcular": lambda d: (d["790"] - d["720"]) / (d["790"] + d["720"]),
    },
    "reip": {
        "bandas": ["670", "700", "740", "780"],
        "rotulo": "REIP (nm)",
        "titulo": "Ponto de Inflexão do Red-Edge (REIP)",
        "calcular": lambda d: 700 + 40 * (((d["670"] + d["780"]) / 2 - d["700"]) /
                                          (d["740"] - d["700"])),
    },
    "dd": {
        "bandas": ["672", "701", "720", "749"],
        "rotulo": "DD = (R749 − R720) − (R701 − R672)",
        "titulo": "Índice de Dupla Diferença (DD)",
        "calcular": lambda d: (d["749"] - d["720"]) - (d["701"] - d["672"]),
    },
    "gndvi": {
        "bandas": ["550", "800"],
        "rotulo": "GNDVI = (R800 − R550) / (R800 + R550)",
        "titulo": "Índice de Vegetação por Diferença Normalizada Verde (GNDVI)",
        "calcular": lambda d: (d["800"] - d["550"]) / (d["800"] + d["550"]),
    },
    "ndmi": {
        "bandas": ["860", "1240"],
        "rotulo": "NDMI = (R860 − R1240) / (R860 + R1240)",
        "titulo": "Índice de Umidade por Diferença Normalizada (NDMI)",
        "calcular": lambda d: (d["860"] - d["1240"]) / (d["860"] + d["1240"]),
    },
    "msi": {
        "bandas": ["820", "1600"],
        "rotulo": "MSI = R1600 / R820",
        "titulo": "Índice de Estresse por Umidade (MSI)",
        "calcular": lambda d: d["1600"] / d["820"],
    },
    "wdi": {
        "bandas": ["900", "970"],
        "rotulo": "WDI = R970 / R900",
        "titulo": "Índice de Déficit Hídrico (WDI)",
        "calcular": lambda d: d["970"] / d["900"],
    },
}


def ordenar_dias(serie: pd.Series) -> pd.Categorical:
    dias = serie.str.extract(r"D(\d+)", expand=False).astype(int)
    return pd.Categorical(dias, categories=sorted(dias.unique()), ordered=True)


def resumir(dados: pd.DataFrame, indice: str) -> pd.DataFrame:
    resumo = (dados.groupby(["genotipo", "condicao", "dia"], observed=True)[indice]
              .agg(media="mean", desvio_padrao="std", n="count").reset_index())
    erro = resumo["desvio_padrao"] / np.sqrt(resumo["n"])
    t_critico = (resumo["n"] - 1).map(T_CRITICO_95).fillna(1.96)
    margem = t_critico * erro
    resumo["ic95_inferior"] = resumo["media"] - margem
    resumo["ic95_superior"] = resumo["media"] + margem
    return resumo.sort_values(["genotipo", "condicao", "dia"])


def gerar_indice(nome: str, config: dict, dados_originais: pd.DataFrame) -> None:
    bandas = config["bandas"]
    dados = dados_originais[["genotipo", "condicao", "data_coleta", *bandas]].copy()
    for banda in bandas:
        dados[banda] = pd.to_numeric(
            dados[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    dados = dados.dropna(subset=bandas)
    with np.errstate(divide="ignore", invalid="ignore"):
        dados[nome] = config["calcular"](dados)
    dados = dados.replace([np.inf, -np.inf], np.nan).dropna(subset=[nome])
    dados["dia"] = ordenar_dias(dados["data_coleta"])
    resumo = resumir(dados, nome)
    resumo.to_csv(ROOT / "resultados" / f"{nome}_por_dia_resumo.csv", index=False, float_format="%.6f")

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
    fig.suptitle(f"{config['titulo']} ao longo dos dias", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.89, "Turno da manhã; linhas = média das leituras e faixas = IC de 95%",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.savefig(ROOT / "resultados" / f"{nome}_por_dia_genotipos.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "resultados" / f"{nome}_por_dia_genotipos.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dados = pd.read_csv(ARQUIVO_ENTRADA, sep=";")
    dados = dados.loc[(dados["turno"].str.lower() == "manha")
                      & dados["genotipo"].isin(GENOTIPOS)
                      & dados["condicao"].isin(CONDICOES)].copy()
    for nome, config in INDICES.items():
        gerar_indice(nome, config, dados)
        print(f"{nome.upper()}: gráfico e resumo salvos.")


if __name__ == "__main__":
    main()
