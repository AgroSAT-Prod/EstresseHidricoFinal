#!/usr/bin/env python3
"""Plota curvas espectrais por bloco, data/turno, genótipo e condição.

Cada figura corresponde a um bloco e contém um painel para cada combinação
de data/turno disponível. Em cada painel são apresentadas as médias das seis
combinações de genótipo (cor) e condição hídrica (estilo de linha).

Uso:
    python3 analiseBlocosCasualizados/plot_curvas_espectrais_genotipos_por_bloco.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
PASTA_SAIDA = Path(__file__).resolve().parent / "curvas_espectrais_genotipos_por_bloco"

BLOCOS = ("B1", "B2", "B3", "B4")
GENOTIPOS = ("BR16", "CD202", "EMB48")
CONDICOES = ("IRRIG", "NIRRIG")
CORES = {"BR16": "#1677c8", "CD202": "#e08b19", "EMB48": "#27864b"}
ESTILOS = {"IRRIG": "-", "NIRRIG": "--"}
ROTULOS_TURNO = {"manha": "Manhã", "tarde": "Tarde"}


def carregar_dados() -> tuple[pd.DataFrame, np.ndarray]:
    dados = pd.read_csv(ENTRADA, sep=";", decimal=",")
    bandas = np.array(sorted(int(col) for col in dados.columns if col.isdigit()))
    if bandas.size == 0:
        raise ValueError("Nenhuma coluna espectral numérica foi encontrada.")
    colunas_bandas = bandas.astype(str)
    dados[colunas_bandas] = dados[colunas_bandas].apply(pd.to_numeric, errors="coerce")
    if dados[colunas_bandas].isna().any().any():
        raise ValueError("Há valores espectrais ausentes ou não numéricos.")
    return dados, bandas


def ordenar_data_turno(dados: pd.DataFrame) -> list[tuple[str, str]]:
    ordem_turno = {"manha": 0, "tarde": 1}
    pares = dados[["data_coleta", "turno"]].drop_duplicates().copy()
    pares["dia_numero"] = pares["data_coleta"].str.extract(r"D(\d+)").astype(int)
    pares["ordem_turno"] = pares["turno"].map(ordem_turno).fillna(99)
    pares = pares.sort_values(["dia_numero", "ordem_turno"])
    return list(pares[["data_coleta", "turno"]].itertuples(index=False, name=None))


def plotar_bloco(dados: pd.DataFrame, bandas: np.ndarray, bloco: str,
                 datas_turnos: list[tuple[str, str]]) -> None:
    n_colunas = 2
    n_linhas = int(np.ceil(len(datas_turnos) / n_colunas))
    fig, eixos = plt.subplots(
        n_linhas, n_colunas, figsize=(14, 3.25 * n_linhas), sharex=True, sharey=True
    )
    eixos = np.asarray(eixos).ravel()
    colunas_bandas = bandas.astype(str)

    for ax, (data_coleta, turno) in zip(eixos, datas_turnos):
        recorte = dados.loc[
            (dados["bloco"] == bloco)
            & (dados["data_coleta"] == data_coleta)
            & (dados["turno"] == turno)
        ]
        for genotipo in GENOTIPOS:
            for condicao in CONDICOES:
                grupo = recorte.loc[
                    (recorte["genotipo"] == genotipo) & (recorte["condicao"] == condicao),
                    colunas_bandas,
                ]
                if not grupo.empty:
                    ax.plot(
                        bandas, grupo.mean(axis=0).to_numpy(),
                        color=CORES[genotipo], linestyle=ESTILOS[condicao], linewidth=1.35,
                    )
        ax.set_title(f"{data_coleta} — {ROTULOS_TURNO.get(turno, turno.title())}",
                     fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in eixos[len(datas_turnos):]:
        ax.set_visible(False)
    for ax in eixos[-n_colunas:]:
        if ax.get_visible():
            ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    for ax in eixos[::n_colunas]:
        ax.set_ylabel("Reflectância", fontsize=9)

    legenda_genotipos = [
        Line2D([0], [0], color=CORES[genotipo], linewidth=2, label=genotipo)
        for genotipo in GENOTIPOS
    ]
    legenda_condicoes = [
        Line2D([0], [0], color="#333333", linestyle=ESTILOS[condicao], linewidth=2,
               label="Irrigado" if condicao == "IRRIG" else "Não irrigado")
        for condicao in CONDICOES
    ]
    fig.legend(
        handles=legenda_genotipos + legenda_condicoes, loc="lower center", ncol=5,
        frameon=False, fontsize=10,
    )
    fig.suptitle(
        f"Curvas espectrais médias por genótipo e condição hídrica — {bloco}\n"
        "Cor: genótipo | linha contínua: irrigado | linha tracejada: não irrigado",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    for extensao in ("pdf", "png"):
        fig.savefig(PASTA_SAIDA / f"curvas_espectrais_genotipos_{bloco}.{extensao}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    dados, bandas = carregar_dados()
    datas_turnos = ordenar_data_turno(dados)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    for bloco in BLOCOS:
        plotar_bloco(dados, bandas, bloco, datas_turnos)
    print(f"Figuras salvas em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
