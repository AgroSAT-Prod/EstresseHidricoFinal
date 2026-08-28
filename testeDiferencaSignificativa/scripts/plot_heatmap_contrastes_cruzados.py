#!/usr/bin/env python3
"""Heatmap dos contrastes cruzados derivados de ANOVA (6 celulas) + Tukey."""

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
ENTRADA = ROOT.parent / "resultados" / "dataset_gerado" / "tukey_contrastes_cruzados_resumo.csv"
SAIDA_PNG = ROOT.parent / "resultados" / "heatmap_anova_tukey_contrastes_cruzados.png"
SAIDA_PDF = ROOT.parent / "resultados" / "heatmap_anova_tukey_contrastes_cruzados.pdf"
N_BANDAS = 2051
PARES = [("BR16", "CD202"), ("BR16", "EMB48"), ("CD202", "EMB48")]
CRUZAMENTOS = [("IRRIG", "NIRRIG"), ("NIRRIG", "IRRIG")]
CMAP = LinearSegmentedColormap.from_list("cruzados", ["#f7f8fa", "#8fc1c4", "#087f8c"])


def rotulo(a: str, cond_a: str, b: str, cond_b: str) -> str:
    condicoes = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
    return f"{a} ({condicoes[cond_a]}) × {b} ({condicoes[cond_b]})"


def gerar() -> None:
    resumo = pd.read_csv(ENTRADA, sep=";")
    dias = sorted(resumo.dia.unique())
    linhas = [(a, cond_a, b, cond_b) for a, b in PARES for cond_a, cond_b in CRUZAMENTOS]
    if len(resumo) != len(dias) * len(linhas):
        raise ValueError("Resumo incompleto: esperado seis contrastes cruzados em cada dia.")
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
    fig, ax = plt.subplots(figsize=(15, 7), layout="constrained")
    imagem = ax.imshow(valores, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            valor = valores[i, j]
            ax.text(j, i, f"{contagens[i, j]:,}\n{valor:.1f}%".replace(",", "."),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if valor >= 55 else "#17212b")
    ax.set_xticks(range(len(dias)), dias, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(linhas)), [rotulo(*linha) for linha in linhas], fontsize=9)
    ax.axhline(1.5, color="white", linewidth=2.2)
    ax.axhline(3.5, color="white", linewidth=2.2)
    ax.set_title("Contrastes cruzados: genótipo irrigado × outro genótipo não irrigado",
                 fontsize=14, fontweight="bold", pad=12)
    barra = fig.colorbar(imagem, ax=ax, pad=0.02, fraction=0.045)
    barra.set_label("Bandas significativas no Tukey HSD (%)", fontsize=10)
    fig.text(0.01, 0.008,
             "ANOVA de uma via com as seis células (3 genótipos × 2 condições), seguida de Tukey HSD; p ≤ 0,05. "
             "Cada célula mostra n/2.051 e a porcentagem de bandas significativas; turno da manhã; sem FDR entre bandas.",
             fontsize=8.5, color="#4e5964")
    fig.savefig(SAIDA_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG salvo em: {SAIDA_PNG}")
    print(f"PDF salvo em: {SAIDA_PDF}")


if __name__ == "__main__":
    gerar()
