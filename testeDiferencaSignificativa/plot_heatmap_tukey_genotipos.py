#!/usr/bin/env python3
"""Heatmap unico do Tukey HSD entre genotipos, dentro de cada condicao.

Painel equivalente ao "A2" de `plot_anova_tukey_genotipos.py`, isolado:
proporcao de bandas com contraste Tukey HSD significativo (p <= 0,05) por dia
(D02-D10), condicao (IRRIG/NIRRIG) e par de genotipos (BR16 x CD202, BR16 x
EMB48, CD202 x EMB48), em percentual das 2.051 bandas.

Uso:
    python plot_heatmap_tukey_genotipos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as exc:
    raise SystemExit("matplotlib nao esta instalado. Execute: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_gerado"
SAIDA_PNG = ROOT / "heatmap_tukey_genotipos.png"
SAIDA_PDF = ROOT / "heatmap_tukey_genotipos.pdf"

N_BANDAS = 2051
CONDICOES = ["IRRIG", "NIRRIG"]
PARES = [("BR16", "CD202"), ("BR16", "EMB48"), ("CD202", "EMB48")]

COR_CONDICAO = {"IRRIG": "#2878b5", "NIRRIG": "#d96d31"}
CMAP = LinearSegmentedColormap.from_list("anova_tukey", ["#f4f6f8", "#91b8da", "#174c7b"])


def formatar(n: int, percentual: float) -> str:
    return f"{n:,}\n{percentual:.1f}%".replace(",", ".")


def gerar() -> None:
    pares = pd.read_csv(DATA / "anova_tukey_genotipos_resumo_pares.csv", sep=";")
    dias = sorted(pares.dia.unique())
    if len(dias) != 7:
        raise ValueError("Esperados sete dias de avaliacao.")

    linhas = [(cond, a, b) for cond in CONDICOES for a, b in PARES]
    valores = np.array([
        [100 * pares[(pares.dia == dia) & (pares.condicao == cond)
                      & (pares.genotipo_a == a) & (pares.genotipo_b == b)]
         .iloc[0].bandas_sig_tukey / N_BANDAS for dia in dias]
        for cond, a, b in linhas
    ])
    contagens = np.array([
        [pares[(pares.dia == dia) & (pares.condicao == cond)
                & (pares.genotipo_a == a) & (pares.genotipo_b == b)]
         .iloc[0].bandas_sig_tukey for dia in dias]
        for cond, a, b in linhas
    ], dtype=int)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 4.0), constrained_layout=True)
    im = ax.imshow(valores, cmap=CMAP, vmin=0, vmax=100, aspect="auto")

    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            cor = "white" if valores[i, j] >= 55 else "#18212b"
            ax.text(j, i, formatar(contagens[i, j], valores[i, j]),
                    ha="center", va="center", fontsize=9, fontweight="bold", color=cor)

    ax.axhline(2.5, color="white", linewidth=2.2)
    ax.set_xticks(range(len(dias)), dias, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(linhas)), [f"{cond} | {a} x {b}" for cond, a, b in linhas], fontsize=9)
    for tick, (cond, _, _) in zip(ax.get_yticklabels(), linhas):
        tick.set_color(COR_CONDICAO[cond])
    ax.set_title("A2. Tukey HSD significativo\n(% das 2.051 bandas)",
                 fontsize=13, fontweight="bold", pad=10)

    barra = fig.colorbar(im, ax=ax, location="right", pad=0.02, fraction=0.045)
    barra.set_label("Bandas significativas (%)", fontsize=11)

    fig.savefig(SAIDA_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG salvo em: {SAIDA_PNG}")
    print(f"PDF salvo em: {SAIDA_PDF}")


if __name__ == "__main__":
    gerar()
