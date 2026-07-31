#!/usr/bin/env python3
"""Painel visual do teste de normalidade (Shapiro-Wilk banda a banda).

Gera uma unica figura PNG com tres paineis que contam a historia do teste:

A) Dispersao: % de bandas normais (q FDR) x n de amostras (eixo x log).
   Um ponto por grupo, cor por nivel de agrupamento. Expoe o achado central:
   a taxa de "normalidade" despenca conforme n cresce -> artefato de poder do
   Shapiro-Wilk, nao propriedade do espectro.

B) Perfil espectral: proporcao de estratos (genotipo x condicao x dia) em que
   cada banda passa como normal, ao longo do comprimento de onda. Mostra ONDE
   no espectro mora a nao-normalidade; regioes de agua/ruido sao sombreadas.

C) Distribuicao de W por agrupamento (boxplot). Mostra a MAGNITUDE do desvio:
   W concentrado em ~0.98-0.99 em todos os niveis -> desvios pequenos, so
   detectados quando n e grande.

Uso:
    python plot_normalidade.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent

SAIDA_DIR = ROOT / "dataset_gerado"
SAIDA = ROOT / "normalidade_painel.png"

ALPHA = 0.05

# Niveis de agrupamento em ordem crescente de granularidade (n decrescente).
# Cor por nivel — paleta qualitativa Dark2 (daltonico-segura), ordem fixa.
AGRUPAMENTOS = ["global", "genotipo", "condicao", "dia", "genotipo_condicao_dia"]
ROTULOS = {
    "global": "global",
    "genotipo": "genotipo",
    "condicao": "condicao",
    "dia": "dia",
    "genotipo_condicao_dia": "genotipo x condicao x dia",
}
CORES = {
    "global": "#404040",
    "genotipo": "#1b9e77",
    "condicao": "#d95f02",
    "dia": "#7570b3",
    "genotipo_condicao_dia": "#e7298a",
}

# Regioes espectrais tipicamente dominadas por absorcao de agua / ruido (nm).
REGIOES_RUIDO = [(1340, 1460), (1790, 1960), (2400, 2450)]


def carregar_resumo() -> pd.DataFrame:
    """Le o resumo (uma linha por grupo, todos os niveis de agrupamento)."""
    resumo = pd.read_csv(SAIDA_DIR / "normalidade_resumo.csv", sep=";")
    print(f"  normalidade_resumo: {len(resumo)} grupos")
    return resumo


def carregar_por_banda() -> pd.DataFrame:
    """Le a proporcao de estratos normais por banda (perfil espectral)."""
    por_banda = pd.read_csv(SAIDA_DIR / "normalidade_por_banda.csv", sep=";")
    por_banda = por_banda.sort_values("banda_nm")
    print(f"  normalidade_por_banda: {len(por_banda)} bandas")
    return por_banda


def carregar_W_por_agrupamento() -> dict[str, np.ndarray]:
    """Coleta os W de todas as bandas/grupos dentro de cada agrupamento."""
    ws: dict[str, np.ndarray] = {}
    for agr in AGRUPAMENTOS:
        arq = SAIDA_DIR / f"normalidade_shapiro_{agr}.csv"
        df = pd.read_csv(arq, sep=";")
        w = df["W"].to_numpy(dtype=float)
        ws[agr] = w[np.isfinite(w)]
        print(f"  {agr}: {len(ws[agr])} valores de W")
    return ws


def plot_dispersao(ax: plt.Axes, resumo: pd.DataFrame) -> None:
    """% bandas normais (q FDR) x n de amostras, cor por agrupamento."""
    for agr in AGRUPAMENTOS:
        sub = resumo[resumo["agrupamento"] == agr]
        if sub.empty:
            continue
        ax.scatter(
            sub["n_amostras"],
            sub["prop_normais_q"] * 100,
            s=70,
            color=CORES[agr],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=ROTULOS[agr],
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("n de amostras no grupo (escala log)", fontsize=9)
    ax.set_ylabel("Bandas normais - q FDR (%)", fontsize=9)
    ax.set_title(
        "A) Normalidade cai conforme n cresce (artefato de poder do teste)",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", title="Agrupamento", title_fontsize=8)


def plot_perfil_espectral(ax: plt.Axes, por_banda: pd.DataFrame) -> None:
    """Proporcao de estratos normais por banda, ao longo do espectro."""
    for ini, fim in REGIOES_RUIDO:
        ax.axvspan(ini, fim, color="gray", alpha=0.12, zorder=0)

    ax.plot(
        por_banda["banda_nm"],
        por_banda["prop_estratos_normais_q"] * 100,
        color=CORES["genotipo_condicao_dia"],
        linewidth=1.3,
        alpha=0.9,
        label="Estratos normais por banda (q FDR)",
    )

    mediana = float(np.nanmedian(por_banda["prop_estratos_normais_q"])) * 100
    ax.axhline(
        mediana,
        color="#404040",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"Mediana ({mediana:.0f}%)",
    )

    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("Estratos normais (%)", fontsize=9)
    ax.set_title(
        "B) Onde a nao-normalidade mora no espectro "
        "(faixas cinza: absorcao de agua / ruido)",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlim(float(por_banda["banda_nm"].min()), float(por_banda["banda_nm"].max()))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_distribuicao_W(ax: plt.Axes, ws: dict[str, np.ndarray]) -> None:
    """Boxplot de W por agrupamento — magnitude do desvio de normalidade."""
    dados = [ws[agr] for agr in AGRUPAMENTOS]
    posicoes = range(1, len(AGRUPAMENTOS) + 1)

    bp = ax.boxplot(
        dados,
        positions=list(posicoes),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
    )
    for caixa, agr in zip(bp["boxes"], AGRUPAMENTOS):
        caixa.set_facecolor(CORES[agr])
        caixa.set_alpha(0.65)
        caixa.set_edgecolor("black")
        caixa.set_linewidth(0.6)

    ax.axhline(
        1.0, color="#1b9e77", linestyle=":", linewidth=1.0, alpha=0.8,
        label="W = 1 (normal perfeita)",
    )

    ax.set_xticks(list(posicoes))
    ax.set_xticklabels(
        [ROTULOS[a].replace(" x ", "\nx ") for a in AGRUPAMENTOS],
        fontsize=7,
    )
    ax.set_ylabel("Estatistica W de Shapiro-Wilk", fontsize=9)
    ax.set_title(
        "C) Desvios sao pequenos (W ~ 0.99) em todos os niveis",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def gerar_painel(
    resumo: pd.DataFrame,
    por_banda: pd.DataFrame,
    ws: dict[str, np.ndarray],
) -> None:
    """Gera o painel completo."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[1.3, 1],
        hspace=0.32,
        wspace=0.22,
    )

    ax_disp = fig.add_subplot(gs[0, 0])
    ax_boxw = fig.add_subplot(gs[0, 1])
    ax_perf = fig.add_subplot(gs[1, :])

    plot_dispersao(ax_disp, resumo)
    plot_distribuicao_W(ax_boxw, ws)
    plot_perfil_espectral(ax_perf, por_banda)

    fig.suptitle(
        "Teste de normalidade (Shapiro-Wilk banda a banda) - "
        "espectro pre-processado / normalizado\n"
        "Somente turno da manha (unico presente nos sete dias)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(SAIDA, dpi=300, bbox_inches="tight")
    print(f"\nPainel salvo em: {SAIDA}")
    print("  Resolucao: 300 DPI")
    print("  Tamanho: 16x11 pol")


def main() -> None:
    print("Gerando painel do teste de normalidade...\n")
    resumo = carregar_resumo()
    por_banda = carregar_por_banda()
    ws = carregar_W_por_agrupamento()
    gerar_painel(resumo, por_banda, ws)
    print("Concluido.")


if __name__ == "__main__":
    main()
