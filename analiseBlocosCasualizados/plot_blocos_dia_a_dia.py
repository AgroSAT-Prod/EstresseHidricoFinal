#!/usr/bin/env python3
"""Grafico dia a dia da media dos blocos sobre os dados brutos (D04M etc.).

Para cada dia de coleta, gera uma figura com um painel por combinacao de
genotipo e condicao. Em cada painel sao mostradas as leituras individuais
(espectros brutos) de todos os blocos, com a media de cada bloco (B1, B2,
B3 e B4) sobreposta.

Diferente de plot_blocos_d04m.py, esta versao usa o arquivo cru
(Unificada13052026.csv) em vez do processado, normaliza a grafia dos blocos
(B1/b1 -> B1) e nao exige exatamente oito leituras por celula.

Uso:
    python analiseBlocosCasualizados/plot_blocos_dia_a_dia.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "dataset" / "Unificada13052026.csv"
SAIDA_DIR = Path(__file__).resolve().parent

GENOTIPOS = ("BR16", "CD202", "EMB48")
CONDICOES = ("IRRIG", "NIRRIG")
BLOCOS = ("B1", "B2", "B3", "B4")
CORES_BLOCO = {
    "B1": "#d62728",
    "B2": "#1f77b4",
    "B3": "#e66101",
    "B4": "#1b9e77",
}


def carregar_dados() -> tuple[pd.DataFrame, np.ndarray]:
    """Carrega o CSV cru e separa metadados das bandas espectrais."""
    dados = pd.read_csv(ENTRADA, sep=";", decimal=",")
    bandas = np.array([col for col in dados.columns if col.isdigit()], dtype=int)
    if len(bandas) == 0:
        raise ValueError("Nenhuma coluna espectral numerica foi encontrada.")

    # O arquivo cru traz os blocos em caixas alta e baixa (B1, b1, ...).
    dados["bloco"] = dados["bloco"].astype(str).str.upper()
    dados = dados[dados["bloco"].isin(BLOCOS)].copy()

    espectros = (
        dados[bandas.astype(str)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    return dados, bandas, espectros


def plotar_dia(dados: pd.DataFrame, espectros: np.ndarray, bandas: np.ndarray,
               dia: str) -> Path:
    """Gera e salva a figura de um dia de coleta."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, eixos = plt.subplots(3, 2, figsize=(15, 14), sharex=True)

    for linha, genotipo in enumerate(GENOTIPOS):
        for coluna, condicao in enumerate(CONDICOES):
            ax = eixos[linha, coluna]
            celula = (
                (dados["genotipo"] == genotipo)
                & (dados["condicao"] == condicao)
            ).to_numpy()

            for bloco in BLOCOS:
                mascara = celula & (dados["bloco"] == bloco).to_numpy()
                if not mascara.any():
                    continue
                curvas = espectros[mascara]

                # Leituras individuais (brutas) em tom claro.
                for curva in curvas:
                    ax.plot(bandas, curva, color=CORES_BLOCO[bloco],
                            linewidth=0.5, alpha=0.18)

                # Media do bloco sobreposta em traco mais grosso.
                media = np.nanmean(curvas, axis=0)
                ax.plot(
                    bandas, media, color=CORES_BLOCO[bloco], linewidth=2.2,
                    label=f"Media {bloco} (n={int(mascara.sum())})",
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
        Line2D([0], [0], color=CORES_BLOCO[bloco], linewidth=2.2,
                label=f"Media {bloco}")
        for bloco in BLOCOS
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(
        f"{dia} — leituras brutas e medias dos blocos\n"
        "Cada painel representa uma combinacao de genotipo e condicao",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    saida = SAIDA_DIR / f"blocos_{dia}_bruto.png"
    fig.savefig(saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return saida


def principal() -> None:
    dados, bandas, espectros = carregar_dados()
    dias = sorted(dados["data_coleta"].unique())
    print(f"Dias encontrados ({len(dias)}): {dias}")

    for dia in dias:
        recorte = dados["data_coleta"] == dia
        saida = plotar_dia(
            dados.loc[recorte].reset_index(drop=True),
            espectros[recorte.to_numpy()],
            bandas,
            dia,
        )
        print(f"Figura salva em: {saida}")


if __name__ == "__main__":
    principal()
