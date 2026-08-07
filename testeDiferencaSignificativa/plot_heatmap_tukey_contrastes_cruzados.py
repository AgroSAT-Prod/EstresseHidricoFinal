#!/usr/bin/env python3
"""Heatmap unico do painel C: contrastes cruzados genotipo x condicao (Tukey).

Replica o painel "C. Tukey HSD: contrastes cruzados entre genotipo e condicao"
de `plot_heatmaps_anova_tukey_completo.py`, isolado, com a fonte dos valores
numericos um pouco maior.

Uso:
    python plot_heatmap_tukey_contrastes_cruzados.py
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
SAIDA_PNG = ROOT / "heatmap_tukey_contrastes_cruzados.png"
SAIDA_PDF = ROOT / "heatmap_tukey_contrastes_cruzados.pdf"

N_BANDAS = 2051
PARES = [("BR16", "CD202"), ("BR16", "EMB48"), ("CD202", "EMB48")]
CRUZAMENTOS = [("IRRIG", "NIRRIG"), ("NIRRIG", "IRRIG")]
CMAP = LinearSegmentedColormap.from_list("anovatukey", ["#f7f8fa", "#8fc1c4", "#087f8c"])


def formatar(n: int, percentual: float) -> str:
    return f"{n:,}\n{percentual:.1f}%".replace(",", ".")


def rotulo(a: str, cond_a: str, b: str, cond_b: str) -> str:
    condicoes = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
    return f"{a} ({condicoes[cond_a]}) × {b} ({condicoes[cond_b]})"


def gerar() -> None:
    resumo = pd.read_csv(DATA / "tukey_contrastes_cruzados_resumo.csv", sep=";")
    dias = sorted(resumo.dia.unique())
    if len(dias) != 7:
        raise ValueError("Esperados sete dias de avaliacao.")

    linhas = [(a, cond_a, b, cond_b)
              for a, b in PARES for cond_a, cond_b in CRUZAMENTOS]
    valores = np.empty((len(linhas), len(dias)))
    contagens = np.empty_like(valores, dtype=int)
    for i, (a, cond_a, b, cond_b) in enumerate(linhas):
        for j, dia in enumerate(dias):
            sub = resumo[(resumo.dia == dia) & (resumo.genotipo_a == a)
                         & (resumo.condicao_a == cond_a) & (resumo.genotipo_b == b)
                         & (resumo.condicao_b == cond_b)]
            if len(sub) != 1:
                raise ValueError(f"Contraste ausente ou duplicado: {dia}, {a}, {cond_a}, {b}, {cond_b}")
            valores[i, j] = 100 * sub.iloc[0].prop_bandas_sig_tukey
            contagens[i, j] = sub.iloc[0].bandas_sig_tukey

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    im = ax.imshow(valores, cmap=CMAP, vmin=0, vmax=100, aspect="auto")

    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            cor = "white" if valores[i, j] >= 55 else "#17212b"
            ax.text(j, i, formatar(contagens[i, j], valores[i, j]),
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color=cor)

    for y in [1.5, 3.5]:
        ax.axhline(y, color="white", linewidth=2.2)
    ax.set_xticks(range(len(dias)), dias, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(linhas)), [rotulo(*linha) for linha in linhas], fontsize=9)
    ax.set_title("Tukey HSD: contrastes cruzados entre genótipo e condição",
                 fontsize=12, fontweight="bold", pad=10)

    barra = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    barra.set_label("Bandas significativas (%)", fontsize=11)

    fig.savefig(SAIDA_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG salvo em: {SAIDA_PNG}")
    print(f"PDF salvo em: {SAIDA_PDF}")


if __name__ == "__main__":
    gerar()
