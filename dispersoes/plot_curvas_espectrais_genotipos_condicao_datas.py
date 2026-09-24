#!/usr/bin/env python3
"""Curvas espectrais médias dos genótipos agrupados por condição e data.

Gera uma figura para 25/02 (D04M) e outra para 02/03 (D09M). Cada figura
compara irrigado (azul) e não irrigado (vermelho); os três genótipos e os
quatro blocos são agrupados em cada curva média.

Uso:
    python dispersoes/plot_curvas_espectrais_genotipos_condicao_datas.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
PASTA_SAIDA = Path(__file__).resolve().parent

DATAS = (("D04M", "25/02"), ("D09M", "02/03"))
CONDICOES = (("IRRIG", "Irrigado"), ("NIRRIG", "Não irrigado"))
CORES = {"IRRIG": "#1976d2", "NIRRIG": "#d32f2f"}


def carregar_dados() -> tuple[pd.DataFrame, np.ndarray]:
    dados = pd.read_csv(ENTRADA, sep=";", decimal=",")
    bandas = np.array(sorted(int(c) for c in dados.columns if c.isdigit()))
    if not len(bandas):
        raise ValueError("Nenhuma banda espectral foi encontrada.")
    colunas = bandas.astype(str).tolist()
    dados[colunas] = dados[colunas].apply(pd.to_numeric, errors="coerce")
    return dados, bandas


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    dados, bandas = carregar_dados()
    colunas = bandas.astype(str).tolist()
    for codigo_data, rotulo_data in DATAS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for condicao, rotulo_condicao in CONDICOES:
            recorte = dados.loc[
                dados["data_coleta"].eq(codigo_data)
                & dados["turno"].eq("manha")
                & dados["condicao"].eq(condicao)
            ]
            if recorte.empty:
                raise ValueError(f"Sem leituras para {codigo_data} / {condicao}.")
            ax.plot(
                bandas, recorte[colunas].mean(axis=0), color=CORES[condicao], linewidth=2.4,
                label=f"{rotulo_condicao} (n={len(recorte)})",
            )
        ax.set_title(
            f"Curvas espectrais médias — genótipos agrupados ({rotulo_data})\n"
            "Leituras matinais reunidas dos três genótipos e quatro blocos",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Comprimento de onda (nm)", fontsize=16)
        ax.set_ylabel("Reflectância média", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.legend(
            frameon=False, title="Condição hídrica", fontsize=14,
            title_fontsize=15,
        )
        fig.tight_layout()
        nome = f"curvas_espectrais_genotipos_agrupados_irr_nirrig_{rotulo_data.replace('/', '_')}"
        for extensao in ("png", "pdf"):
            saida = PASTA_SAIDA / f"{nome}.{extensao}"
            fig.savefig(saida, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Figura salva em: {PASTA_SAIDA / (nome + '.png')}")

    # Calcula primeiro a média de cada dia e só então a média entre os dias,
    # garantindo o mesmo peso para cada um dos sete dias de coleta.
    datas_media = sorted(
        dados.loc[dados["turno"].eq("manha"), "data_coleta"].dropna().unique(),
        key=lambda data: int(data[1:3]),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for condicao, rotulo_condicao in CONDICOES:
        medias_diarias = []
        for codigo_data in datas_media:
            recorte = dados.loc[
                dados["data_coleta"].eq(codigo_data)
                & dados["turno"].eq("manha")
                & dados["condicao"].eq(condicao),
                colunas,
            ]
            if recorte.empty:
                raise ValueError(f"Sem leituras para {codigo_data} / {condicao}.")
            medias_diarias.append(recorte.mean(axis=0).to_numpy())
        ax.plot(
            bandas, np.mean(medias_diarias, axis=0), color=CORES[condicao], linewidth=2.4,
            label=rotulo_condicao,
        )
    ax.set_title(
        "Curvas espectrais médias — genótipos agrupados\n"
        "Média dos sete dias de coleta; leituras matinais dos quatro blocos",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=16)
    ax.set_ylabel("Reflectância média", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, title="Condição hídrica", fontsize=14, title_fontsize=15)
    fig.tight_layout()
    nome = "curvas_espectrais_genotipos_agrupados_irr_nirrig_media_todos_dias"
    for extensao in ("png", "pdf"):
        fig.savefig(PASTA_SAIDA / f"{nome}.{extensao}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {PASTA_SAIDA / (nome + '.png')}")


if __name__ == "__main__":
    main()
