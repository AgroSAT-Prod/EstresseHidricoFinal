#!/usr/bin/env python3
"""Reencode do achado do teste de normalidade via distribuicao dos p-valores.

Mesma informacao do painel de dispersao (% normal cai conforme n cresce),
mostrada pelo mecanismo por tras dela: a distribuicao dos p-valores do
Shapiro-Wilk.

Ideia central: sob normalidade VERDADEIRA os p-valores seriam uniformes em
[0, 1] (histograma plano; ECDF na diagonal). Quanto MAIS amostras o grupo tem,
mais o Shapiro detecta desvios minusculos -> os p-valores se empilham perto de
zero. O grau desse empilhamento e a "nao-normalidade" observada.

A figura tem dois blocos:
- 5 histogramas de p-valor, um por nivel de agrupamento, ordenados por n
  (do maior para o menor). A linha tracejada marca o patamar uniforme esperado.
- 1 painel ECDF sobrepondo os 5 niveis, com a diagonal de referencia (uniforme).

Uso:
    python plot_normalidade_pvalores.py
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
SAIDA = ROOT / "normalidade_pvalores.png"

ALPHA = 0.05
N_BINS = 20

# Niveis ordenados por n de amostras por grupo (decrescente): quanto maior n,
# mais os p-valores colapsam para zero.
AGRUPAMENTOS = ["global", "condicao", "genotipo", "dia", "genotipo_condicao_dia"]
ROTULOS = {
    "global": "global",
    "condicao": "condicao",
    "genotipo": "genotipo",
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


def carregar_n_tipico() -> dict[str, int]:
    """n mediano por grupo em cada agrupamento (lido do resumo, nao fixado)."""
    resumo = pd.read_csv(SAIDA_DIR / "normalidade_resumo.csv", sep=";")
    mediana = resumo.groupby("agrupamento")["n_amostras"].median()
    return {agr: int(round(mediana[agr])) for agr in AGRUPAMENTOS}


def carregar_pvalores() -> dict[str, np.ndarray]:
    """Coleta os p-valores de todas as bandas/grupos por agrupamento."""
    pvals: dict[str, np.ndarray] = {}
    for agr in AGRUPAMENTOS:
        df = pd.read_csv(SAIDA_DIR / f"normalidade_shapiro_{agr}.csv", sep=";")
        p = df["p_valor"].to_numpy(dtype=float)
        pvals[agr] = p[np.isfinite(p)]
        print(f"  {agr}: {len(pvals[agr])} p-valores")
    return pvals


def plot_histograma(ax: plt.Axes, agr: str, p: np.ndarray, n_tipico: int) -> None:
    """Histograma de p-valores de um agrupamento com referencia uniforme."""
    ax.hist(
        p,
        bins=N_BINS,
        range=(0, 1),
        color=CORES[agr],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )

    # Patamar esperado sob normalidade verdadeira (distribuicao uniforme).
    esperado = len(p) / N_BINS
    ax.axhline(
        esperado,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Esperado se normal (uniforme)",
    )
    ax.axvline(ALPHA, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.8)

    prop_normal = float(np.mean(p > ALPHA)) * 100
    ax.set_title(
        f"{ROTULOS[agr]}\nn~{n_tipico}  |  p>0.05: {prop_normal:.0f}%",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("p-valor (Shapiro-Wilk)", fontsize=8)
    ax.set_ylabel("N. de bandas", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    if agr == AGRUPAMENTOS[0]:
        ax.legend(fontsize=6.5, loc="upper right")


def plot_ecdf(
    ax: plt.Axes,
    pvals: dict[str, np.ndarray],
    n_tipico: dict[str, int],
) -> None:
    """ECDF dos p-valores por nivel, com a diagonal uniforme de referencia."""
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0,
            alpha=0.6, label="Uniforme (se normal)")

    for agr in AGRUPAMENTOS:
        p = np.sort(pvals[agr])
        y = np.arange(1, len(p) + 1) / len(p)
        ax.plot(p, y, color=CORES[agr], linewidth=1.6, alpha=0.9,
                label=f"{ROTULOS[agr]} (n~{n_tipico[agr]})")

    ax.axvline(ALPHA, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title(
        "ECDF dos p-valores: quanto mais alta a curva perto de 0, "
        "mais rejeicoes (efeito de n)",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("p-valor (Shapiro-Wilk)", fontsize=8)
    ax.set_ylabel("Proporcao acumulada de bandas", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")


def gerar_painel(pvals: dict[str, np.ndarray], n_tipico: dict[str, int]) -> None:
    """Gera a figura: 5 histogramas + 1 ECDF."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    posicoes = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for agr, (i, j) in zip(AGRUPAMENTOS, posicoes):
        ax = fig.add_subplot(gs[i, j])
        plot_histograma(ax, agr, pvals[agr], n_tipico[agr])

    ax_ecdf = fig.add_subplot(gs[1, 2])
    plot_ecdf(ax_ecdf, pvals, n_tipico)

    fig.suptitle(
        "Distribuicao dos p-valores do Shapiro-Wilk por nivel de agrupamento "
        "(somente turno da manha)\n"
        "Sob normalidade verdadeira seriam uniformes; o empilhamento em p~0 "
        "cresce com o n -> nao-normalidade e efeito de poder do teste",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )

    plt.savefig(SAIDA, dpi=300, bbox_inches="tight")
    print(f"\nPainel salvo em: {SAIDA}")
    print("  Resolucao: 300 DPI")
    print("  Tamanho: 16x9 pol")


def main() -> None:
    print("Gerando painel de p-valores do teste de normalidade...\n")
    pvals = carregar_pvalores()
    n_tipico = carregar_n_tipico()
    gerar_painel(pvals, n_tipico)
    print("Concluido.")


if __name__ == "__main__":
    main()
