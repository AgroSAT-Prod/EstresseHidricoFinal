#!/usr/bin/env python3
"""Painel ANOVA + Tukey por dia, condicao e genotipos.

O painel A resume a proporcao de bandas significativas; o painel B mostra
onde essas diferencas aparecem no espectro. A analise de origem usa ANOVA de
uma via e Tukey HSD com p <= 0,05, sem FDR entre as 2.051 bandas.

Uso:
    python3 plot_anova_tukey_genotipos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
except ImportError as exc:
    raise SystemExit("matplotlib nao esta instalado. Execute: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset_gerado"
SAIDA_PNG = ROOT / "anova_tukey_genotipos_painel.png"
SAIDA_PDF = ROOT / "anova_tukey_genotipos_painel.pdf"

ALPHA = 0.05
N_BANDAS = 2051
CONDICOES = ["IRRIG", "NIRRIG"]
PARES = [("BR16", "CD202"), ("BR16", "EMB48"), ("CD202", "EMB48")]

COR_CONDICAO = {"IRRIG": "#2878b5", "NIRRIG": "#d96d31"}
COR_PAR = {
    ("BR16", "CD202"): "#2a9d8f",
    ("BR16", "EMB48"): "#e76f51",
    ("CD202", "EMB48"): "#6d5bd0",
}
CMAP = LinearSegmentedColormap.from_list("anova_tukey", ["#f4f6f8", "#91b8da", "#174c7b"])
REGIOES = [("VIS", 400, 700), ("RE", 700, 780), ("NIR", 780, 1350),
           ("SWIR1", 1350, 1800), ("SWIR2", 1800, 2450)]
Y_MAX = 12


def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Le os resultados e confirma a estrutura esperada do delineamento."""
    anova = pd.read_csv(DATA_DIR / "anova_genotipos_por_dia.csv", sep=";")
    tukey = pd.read_csv(DATA_DIR / "tukey_genotipos_por_dia.csv", sep=";")
    resumo = pd.read_csv(DATA_DIR / "anova_tukey_genotipos_resumo.csv", sep=";")
    pares = pd.read_csv(DATA_DIR / "anova_tukey_genotipos_resumo_pares.csv", sep=";")

    dias = sorted(anova["dia"].unique())
    esperado = len(dias) * len(CONDICOES) * N_BANDAS
    if len(anova) != esperado:
        raise ValueError(f"ANOVA com {len(anova)} linhas; esperado: {esperado}.")
    if len(resumo) != len(dias) * len(CONDICOES):
        raise ValueError("Resumo ANOVA nao contem todas as combinacoes dia x condicao.")
    if len(pares) != len(dias) * len(CONDICOES) * len(PARES):
        raise ValueError("Resumo Tukey nao contem os tres pares por dia e condicao.")
    if not (tukey["significativo_tukey"] <= tukey["p_anova"].le(ALPHA)).all():
        raise ValueError("Ha contraste Tukey significativo sem ANOVA significativa.")
    return anova, tukey, resumo, pares


def anotar_heatmap(ax: plt.Axes, valores: np.ndarray, contagens: np.ndarray) -> None:
    """Anota contagem e porcentagem, escolhendo texto legivel pelo fundo."""
    for linha in range(valores.shape[0]):
        for coluna in range(valores.shape[1]):
            valor = valores[linha, coluna]
            cor = "white" if valor >= 55 else "#18212b"
            ax.text(coluna, linha, f"{contagens[linha, coluna]:,}\n{valor:.1f}%".replace(",", "."),
                    ha="center", va="center", fontsize=7.2, fontweight="bold", color=cor)


def painel_resumo(fig: plt.Figure, spec, resumo: pd.DataFrame, pares: pd.DataFrame, dias: list[str]) -> None:
    """Dois heatmaps: ANOVA global e Tukey por par."""
    sub = spec.subgridspec(1, 2, width_ratios=[1, 2.75], wspace=0.18)
    ax_anova = fig.add_subplot(sub[0, 0])
    ax_tukey = fig.add_subplot(sub[0, 1])

    valores_anova = np.array([
        [100 * resumo[(resumo.dia == dia) & (resumo.condicao == cond)].iloc[0].prop_bandas_sig_anova
         for dia in dias]
        for cond in CONDICOES
    ])
    contagens_anova = np.array([
        [resumo[(resumo.dia == dia) & (resumo.condicao == cond)].iloc[0].bandas_sig_anova
         for dia in dias]
        for cond in CONDICOES
    ], dtype=int)
    im = ax_anova.imshow(valores_anova, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    anotar_heatmap(ax_anova, valores_anova, contagens_anova)
    ax_anova.set_xticks(range(len(dias)), dias, fontsize=8)
    ax_anova.set_yticks(range(len(CONDICOES)), CONDICOES, fontsize=8, fontweight="bold")
    for tick, cond in zip(ax_anova.get_yticklabels(), CONDICOES):
        tick.set_color(COR_CONDICAO[cond])
    ax_anova.set_title("A1. ANOVA significativa\n(% das 2.051 bandas)", fontsize=10, fontweight="bold")

    linhas = [(cond, par) for cond in CONDICOES for par in PARES]
    valores_tukey = np.array([
        [100 * pares[(pares.dia == dia) & (pares.condicao == cond)
                      & (pares.genotipo_a == par[0]) & (pares.genotipo_b == par[1])]
         .iloc[0].bandas_sig_tukey / N_BANDAS for dia in dias]
        for cond, par in linhas
    ])
    contagens_tukey = np.array([
        [pares[(pares.dia == dia) & (pares.condicao == cond)
                & (pares.genotipo_a == par[0]) & (pares.genotipo_b == par[1])]
         .iloc[0].bandas_sig_tukey for dia in dias]
        for cond, par in linhas
    ], dtype=int)
    ax_tukey.imshow(valores_tukey, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    anotar_heatmap(ax_tukey, valores_tukey, contagens_tukey)
    ax_tukey.set_xticks(range(len(dias)), dias, fontsize=8)
    rotulos = [f"{cond} | {a} x {b}" for cond, (a, b) in linhas]
    ax_tukey.set_yticks(range(len(linhas)), rotulos, fontsize=7.3)
    for tick, (cond, _) in zip(ax_tukey.get_yticklabels(), linhas):
        tick.set_color(COR_CONDICAO[cond])
    ax_tukey.axhline(2.5, color="white", linewidth=2.2)
    ax_tukey.set_title("A2. Tukey HSD significativo\n(% das 2.051 bandas)", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=[ax_anova, ax_tukey], location="right", pad=0.018,
                 fraction=0.04, label="Bandas significativas (%)")


def painel_espectral(fig: plt.Figure, spec, anova: pd.DataFrame, tukey: pd.DataFrame, dias: list[str]) -> None:
    """Grade 2 x 7 de significancia ANOVA e faixas Tukey ao longo do espectro."""
    grade = spec.subgridspec(2, len(dias), hspace=0.18, wspace=0.12)
    for i, cond in enumerate(CONDICOES):
        for j, dia in enumerate(dias):
            ax = fig.add_subplot(grade[i, j])
            sub_anova = anova[(anova.dia == dia) & (anova.condicao == cond)].sort_values("banda_nm")
            w = sub_anova.banda_nm.to_numpy(float)
            p = sub_anova.p_anova.to_numpy(float)
            with np.errstate(divide="ignore"):
                y = -np.log10(np.maximum(p, np.finfo(float).tiny))
            y = np.clip(y, 0, Y_MAX)
            sig_anova = sub_anova.significativo_anova.to_numpy(bool)

            ax.fill_between(w, 0, Y_MAX, where=sig_anova, color="#dbe8f2", step="mid", linewidth=0)
            ax.plot(w, y, color="#17212b", linewidth=0.65, zorder=3)
            ax.axhline(-np.log10(ALPHA), color="#b64e2d", linestyle="--", linewidth=0.8, zorder=4)

            sub_tukey = tukey[(tukey.dia == dia) & (tukey.condicao == cond)]
            for faixa, par in enumerate(PARES):
                valores = sub_tukey[(sub_tukey.genotipo_a == par[0]) & (sub_tukey.genotipo_b == par[1])]
                mascara = (valores.set_index("banda_nm").reindex(w).significativo_tukey
                           .fillna(False).to_numpy(bool))
                inicio, fim = 0.12 + faixa * 0.27, 0.33 + faixa * 0.27
                ax.fill_between(w, inicio, fim, where=mascara, step="mid", linewidth=0,
                                color=COR_PAR[par], zorder=5)

            for _, inicio, _ in REGIOES[1:]:
                ax.axvline(inicio, color="#76808a", linewidth=0.45, alpha=0.65, zorder=2)
            if i == 0:
                ax.set_title(dia, fontsize=9, fontweight="bold", pad=6)
            if i == 1:
                ax.set_xlabel("nm", fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{cond}\n−log10(p)", fontsize=8, fontweight="bold", color=COR_CONDICAO[cond])
            else:
                ax.set_yticklabels([])
            prop = 100 * sig_anova.mean()
            ax.text(0.97, 0.95, f"{prop:.1f}%", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.2, fontweight="bold", color="#26313b")
            ax.set_xlim(w.min(), w.max())
            ax.set_ylim(0, Y_MAX)
            ax.set_yticks([0, 4, 8, 12])
            ax.tick_params(labelsize=6.5, length=2)
            ax.grid(axis="y", alpha=0.23, linewidth=0.5)


def gerar() -> None:
    anova, tukey, resumo, pares = carregar()
    dias = sorted(anova.dia.unique())
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(19, 15.5), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 3.6], hspace=0.29,
                             left=0.07, right=0.93, top=0.91, bottom=0.07)
    painel_resumo(fig, outer[0], resumo, pares, dias)
    painel_espectral(fig, outer[1], anova, tukey, dias)

    fig.suptitle("Diferenças espectrais entre genótipos por dia e condição",
                 fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.07, 0.935,
             "ANOVA de uma via seguida de Tukey HSD, p ≤ 0,05; turno da manhã; 2.051 bandas (400–2450 nm).",
             fontsize=9.2, color="#4e5964")
    fig.legend(handles=[
        Patch(facecolor="#dbe8f2", label="ANOVA significativa"),
        Patch(facecolor=COR_PAR[("BR16", "CD202")], label="Tukey: BR16 × CD202"),
        Patch(facecolor=COR_PAR[("BR16", "EMB48")], label="Tukey: BR16 × EMB48"),
        Patch(facecolor=COR_PAR[("CD202", "EMB48")], label="Tukey: CD202 × EMB48"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.007))
    fig.text(0.07, 0.028,
             "Painel B: faixas coloridas na base indicam os pares significativos no Tukey; a linha tracejada marca p = 0,05.\n"
             "Os valores nos heatmaps e nos cantos dos gráficos são percentuais sobre as 2.051 bandas; sem correção FDR entre bandas.",
             fontsize=8, color="#4e5964", va="bottom")
    fig.savefig(SAIDA_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG salvo em: {SAIDA_PNG}")
    print(f"PDF salvo em: {SAIDA_PDF}")


if __name__ == "__main__":
    gerar()
