#!/usr/bin/env python3
"""Curvas BR16/D02: espectro completo versus redução Spearman, manhã."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))

from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

SAIDA = ROOT / "analiseBR16PorDia" / "curvas_BR16_D02_spearman.png"
TOP5 = (
    ROOT / "analiseBR16PorDia" / "dataset_gerado" / "BR16"
    / "D02" / "top5_bandas.csv"
)


def main() -> None:
    meta, X, w = carregar("normalizado", turno="manha")
    mask = (meta["genotipo"].eq("BR16") & meta["dia"].eq("D02")).to_numpy()
    meta, X = meta.loc[mask].reset_index(drop=True), X[mask]
    condicao = meta["condicao"].to_numpy()

    medias = {
        nome: X[condicao == nome].mean(axis=0)
        for nome in ("IRRIG", "NIRRIG")
    }
    grupos = agrupar(spearman_matriz(X), w)
    reps = np.array([
        np.flatnonzero(grupos == g)[0] for g in np.unique(grupos)
    ])
    top5 = pd.read_csv(TOP5, sep=";").sort_values("posicao")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharey=True, constrained_layout=True)
    cores = {"IRRIG": "#2878B5", "NIRRIG": "#C63D4B"}

    for nome, media in medias.items():
        axes[0].plot(w, media, color=cores[nome], lw=1.4, label=nome)
    axes[0].set_title("BR16 — D02, manhã: curvas médias completas (2.051 bandas)", loc="left", weight="bold")
    axes[0].legend(title="Condição", ncol=2, frameon=True)

    for nome, media in medias.items():
        axes[1].plot(w[reps], media[reps], color=cores[nome], lw=0.8, alpha=0.55)
        axes[1].scatter(w[reps], media[reps], color=cores[nome], s=10, alpha=0.75,
                        label=f"{nome} — representantes")
    for _, linha in top5.iterrows():
        banda = int(linha["banda_nm"])
        idx = int(np.where(w.astype(int) == banda)[0][0])
        axes[1].axvline(banda, color="#E69F00", lw=1.0, alpha=0.8, zorder=1)
        for nome, media in medias.items():
            axes[1].scatter(banda, media[idx], s=78, marker="o", color="#E69F00",
                            edgecolor="black", linewidth=0.7, zorder=5)
    axes[1].set_title(
        "Após Spearman |ρ| > 0,80: 206 bandas representantes; Top 5 PLS-DA em amarelo",
        loc="left", weight="bold",
    )
    axes[1].text(
        0.015, 0.06,
        "Top 5 PLS-DA (nm): " + ", ".join(str(int(v)) for v in top5["banda_nm"]),
        transform=axes[1].transAxes, fontsize=9, color="#805400", weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF4CC", "edgecolor": "#E69F00"},
    )

    for ax in axes:
        ax.set_xlim(w.min(), w.max())
        ax.set_ylabel("Reflectância SNV")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Comprimento de onda (nm)")

    fig.savefig(SAIDA, dpi=220, bbox_inches="tight")
    print(f"Figura salva em {SAIDA}")


if __name__ == "__main__":
    main()
