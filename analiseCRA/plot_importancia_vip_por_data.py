#!/usr/bin/env python3
"""Gráficos de importância VIP dos modelos CRA otimizados, por coleta."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "analiseCRA" / "resultados"
COLETAS = {"24_02": "24/02", "02_03": "02/03", "03_03": "03/03"}


def plotar(codigo: str, data: str) -> None:
    dados = pd.read_csv(RESULTADOS / f"{codigo}_bandas_vip_ge_1.csv", sep=";")
    dados = dados.loc[dados["banda_nm"] >= 400].nlargest(5, "vip_modelo_final").sort_values("vip_modelo_final")
    fig, ax = plt.subplots(figsize=(8.5, max(5.5, len(dados) * .32)))
    ax.barh(dados["banda_nm"].astype(str), dados["vip_modelo_final"], color="#287a8e")
    ax.axvline(1, color="#bd3f32", ls="--", lw=1.2, label="VIP = 1,0")
    ax.set(xlabel="Importância VIP", ylabel="Banda (nm)", title=f"CRA em {data}: importância das bandas VIP")
    ax.xaxis.label.set_fontsize(16); ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(16); ax.yaxis.label.set_fontweight("bold")
    ax.grid(axis="x", alpha=.22)
    ax.legend(frameon=True, loc="lower right", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"{codigo}_importancia_vip.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTADOS / f"{codigo}_importancia_vip.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for codigo, data in COLETAS.items():
        plotar(codigo, data)
