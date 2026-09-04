#!/usr/bin/env python3
"""Plota curvas espectrais por bloco, data/turno e condição hídrica.

Para cada bloco é criado um PDF e um PNG com um painel por combinação de
data e turno disponível. Cada curva é a reflectância média de todas as
leituras daquele bloco e condição (os três genótipos são agregados).

Uso:
    python3 analiseBlocosCasualizados/plot_curvas_espectrais_por_bloco.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
PASTA_SAIDA = Path(__file__).resolve().parent / "curvas_espectrais_por_bloco"

BLOCOS = ("B1", "B2", "B3", "B4")
CONDICOES = ("IRRIG", "NIRRIG")
CORES = {"IRRIG": "#1677c8", "NIRRIG": "#d95032"}
ROTULOS = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
ROTULOS_TURNO = {"manha": "Manhã", "tarde": "Tarde"}


def ordenar_data_turno(dados: pd.DataFrame) -> list[tuple[str, str]]:
    """Obtém as combinações existentes, ordenadas por dia e turno."""
    ordem_turno = {"manha": 0, "tarde": 1}
    pares = dados[["data_coleta", "turno"]].drop_duplicates().copy()
    pares["dia_numero"] = pares["data_coleta"].str.extract(r"D(\d+)").astype(int)
    pares["ordem_turno"] = pares["turno"].map(ordem_turno).fillna(99)
    pares = pares.sort_values(["dia_numero", "ordem_turno"])
    return list(pares[["data_coleta", "turno"]].itertuples(index=False, name=None))


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


def plotar_bloco(dados: pd.DataFrame, bandas: np.ndarray, bloco: str,
                 datas_turnos: list[tuple[str, str]]) -> None:
    n_paineis = len(datas_turnos)
    n_colunas = 2
    n_linhas = int(np.ceil(n_paineis / n_colunas))
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
        for condicao in CONDICOES:
            grupo = recorte.loc[recorte["condicao"] == condicao, colunas_bandas]
            if grupo.empty:
                continue
            ax.plot(
                bandas,
                grupo.mean(axis=0).to_numpy(),
                color=CORES[condicao],
                linewidth=1.5,
                label=ROTULOS[condicao],
            )

        ax.set_title(f"{data_coleta} — {ROTULOS_TURNO.get(turno, turno.title())}",
                     fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in eixos[n_paineis:]:
        ax.set_visible(False)
    for ax in eixos[-n_colunas:]:
        if ax.get_visible():
            ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    for ax in eixos[::n_colunas]:
        ax.set_ylabel("Reflectância", fontsize=9)

    handles, labels = eixos[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(
        f"Curvas espectrais médias por condição hídrica — {bloco}\n"
        "Média dos genótipos e subamostras disponíveis em cada data/turno",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    for extensao in ("pdf", "png"):
        fig.savefig(PASTA_SAIDA / f"curvas_espectrais_{bloco}.{extensao}",
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
    print("Painéis:", ", ".join(f"{data}/{turno}" for data, turno in datas_turnos))


if __name__ == "__main__":
    main()
