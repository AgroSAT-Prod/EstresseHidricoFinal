#!/usr/bin/env python3
"""Compara leituras do bloco B1 com medias dos demais blocos no D04M.

Para cada combinacao de genotipo e condicao, mostra as oito subamostras
espectrais do B1 e as curvas medias de B2, B3 e B4. As medias sao calculadas
somente entre leituras que pertencem a mesma combinacao experimental.

Uso:
    python analiseBlocosCasualizados/plot_blocos_d04m.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
SAIDA = Path(__file__).resolve().parent / "blocos_D04M_B1_leituras_e_medias.png"

BLOCO_FOCAL = "B1"
BLOCOS_REFERENCIA = ("B2", "B3", "B4")
DATA_COLETA = "D04M"
TURNO = "manha"
GENOTIPOS = ("BR16", "CD202", "EMB48")
CONDICOES = ("IRRIG", "NIRRIG")
CORES_REFERENCIA = {"B2": "#1f77b4", "B3": "#e66101", "B4": "#1b9e77"}


def carregar_dados() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Carrega o CSV, filtra D04M/manhã e separa metadados das bandas."""
    dados = pd.read_csv(ENTRADA, sep=";", decimal=",")
    bandas = np.array([col for col in dados.columns if col.isdigit()], dtype=int)
    if len(bandas) == 0:
        raise ValueError("Nenhuma coluna espectral numerica foi encontrada.")

    filtro = (dados["data_coleta"] == DATA_COLETA) & (dados["turno"] == TURNO)
    dados = dados.loc[filtro].copy()
    if dados.empty:
        raise ValueError(f"Nenhuma leitura encontrada para {DATA_COLETA}/{TURNO}.")

    espectros = dados[bandas.astype(str)].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(espectros).all():
        raise ValueError("Ha valores espectrais ausentes ou nao numericos no recorte.")
    return dados, bandas, espectros


def validar_delineamento(dados: pd.DataFrame) -> None:
    """Garante as oito subamostras esperadas por celula experimental."""
    contagens = dados.groupby(["genotipo", "condicao", "bloco"]).size()
    esperado = {
        (genotipo, condicao, bloco)
        for genotipo in GENOTIPOS
        for condicao in CONDICOES
        for bloco in (BLOCO_FOCAL, *BLOCOS_REFERENCIA)
    }
    encontrado = set(contagens.index)
    if encontrado != esperado or not (contagens == 8).all():
        raise ValueError(
            "O recorte nao contem exatamente oito leituras para cada "
            "genotipo × condicao × bloco esperado."
        )


def plotar() -> None:
    dados, bandas, espectros = carregar_dados()
    validar_delineamento(dados)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, eixos = plt.subplots(3, 2, figsize=(15, 14), sharex=True)

    for linha, genotipo in enumerate(GENOTIPOS):
        for coluna, condicao in enumerate(CONDICOES):
            ax = eixos[linha, coluna]
            grupo = (dados["genotipo"] == genotipo) & (dados["condicao"] == condicao)
            focal = (grupo & (dados["bloco"] == BLOCO_FOCAL)).to_numpy()

            # As subamostras ficam discretas para que as tres medias de bloco
            # continuem legiveis, inclusive nos trechos de maior sobreposicao.
            for curva in espectros[focal]:
                ax.plot(bandas, curva, color="#606060", linewidth=0.55, alpha=0.28)

            for bloco in BLOCOS_REFERENCIA:
                referencia = (grupo & (dados["bloco"] == bloco)).to_numpy()
                ax.plot(
                    bandas,
                    espectros[referencia].mean(axis=0),
                    color=CORES_REFERENCIA[bloco],
                    linewidth=1.7,
                    label=f"Media {bloco} (n=8)",
                )

            ax.set_title(f"{genotipo} — {condicao}", fontsize=11, fontweight="bold")
            ax.set_xlim(bandas.min(), bandas.max())
            ax.grid(True, alpha=0.28)
            ax.tick_params(labelsize=8)
            if coluna == 0:
                ax.set_ylabel("Reflectancia", fontsize=9)
            if linha == len(GENOTIPOS) - 1:
                ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)

    legenda = [
        Line2D([0], [0], color="#606060", linewidth=1.2, alpha=0.65,
               label="Leituras individuais B1 (n=8)"),
        *[
            Line2D([0], [0], color=CORES_REFERENCIA[bloco], linewidth=2,
                   label=f"Media {bloco} (n=8)")
            for bloco in BLOCOS_REFERENCIA
        ],
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(
        "D04M — leituras do bloco B1 e medias dos blocos de referencia\n"
        "Cada painel representa uma combinacao de genotipo e condicao",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight")
    print(f"Figura salva em: {SAIDA}")
    print(f"Leituras usadas: {len(dados)} ({DATA_COLETA}, {TURNO})")


if __name__ == "__main__":
    plotar()
