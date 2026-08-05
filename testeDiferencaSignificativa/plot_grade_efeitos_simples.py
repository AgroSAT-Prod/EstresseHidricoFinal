#!/usr/bin/env python3
"""Grade completa dos efeitos simples: 3 pares x 2 condicoes x 7 dias.

Uma celula por combinacao testada -- as 42 familias de FDR de
`efeitos_simples.csv`, todas de uma vez.

Cada celula mostra, ao longo do espectro:

- a curva de -log10 do q-valor de Benjamini-Hochberg, banda a banda. Quanto
  mais alta, mais forte a evidencia de que os dois genotipos daquele par
  diferem naquela banda, naquela condicao e naquele dia;
- tres linhas horizontais nos limiares 0.05, 0.01 e 0.001, que e onde a curva
  precisa passar para a banda contar como significativa;
- as regioes espectrais significativas sombreadas em tres tons encaixados, um
  por limiar. Quanto mais escuro, mais rigoroso o criterio que a regiao
  sobrevive.

A leitura util nao e o valor de uma banda isolada, e sim a **largura e a
posicao** das faixas sombreadas: elas dizem em que parte do espectro os
genotipos se separam, e o quanto essa separacao encolhe conforme o limiar
aperta.

A curva satura no topo: o maximo de -log10(q) no conjunto e 16.5, e o eixo vai
a 10. Nao ha perda de leitura -- acima de 10 a diferenca entre um q e outro nao
muda nenhuma decisao, e as faixas sombreadas ja marcam onde ela esta.

Uma grade estendida, com os contrastes em que genotipo e condicao mudam ao
mesmo tempo, esta em `plot_grade_celulas_cruzadas.py`.

Uso:
    python plot_grade_efeitos_simples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from interacao_genotipo_condicao import ALPHAS  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"
SAIDA = ROOT / "grade_efeitos_simples.png"

PARES = [("BR16", "CD202"), ("BR16", "EMB48"), ("CD202", "EMB48")]
CONDICOES = ["IRRIG", "NIRRIG"]

COR_CONDICAO = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}

# Tons encaixados, um por limiar: mais escuro = criterio mais rigoroso.
TOM_ALPHA = {0.05: "#dbe6f4", 0.01: "#a9c6e8", 0.001: "#6a9bd1"}

COR_CURVA = "#1a1a19"
COR_LIMIAR = "#c2410c"
COR_TEXTO_FRACO = "#52514e"

Y_MAX = 10.0

NOME_TESTE = {
    "mannwhitney": "Mann-Whitney exato",
    "kruskal": "Kruskal-Wallis",
}

REGIOES = [("VIS", 400, 700), ("RE", 700, 780), ("NIR", 780, 1350),
           ("SWIR1", 1350, 1800), ("SWIR2", 1800, 2451)]


def desenhar_celula(
    ax: plt.Axes,
    w: np.ndarray,
    q: np.ndarray,
    mostrar_x: bool,
    mostrar_y: bool,
) -> None:
    """Curva de -log10(q) com faixas significativas e limiares."""
    with np.errstate(divide="ignore"):
        y = -np.log10(np.where(q > 0, q, np.nan))
    y = np.clip(np.nan_to_num(y, nan=0.0), 0, Y_MAX)

    # Do limiar mais frouxo para o mais rigoroso, para os tons se encaixarem.
    for alpha in sorted(ALPHAS, reverse=True):
        ax.fill_between(w, 0, Y_MAX, where=q < alpha, step="mid",
                        color=TOM_ALPHA[alpha], linewidth=0, zorder=1)

    for alpha in ALPHAS:
        ax.axhline(-np.log10(alpha), color=COR_LIMIAR, linestyle="--",
                   linewidth=0.8, alpha=0.85, zorder=3)

    ax.plot(w, y, color=COR_CURVA, linewidth=0.7, zorder=4)

    prop = float((q < ALPHAS[0]).mean()) * 100
    ax.text(0.97, 0.93, f"{prop:.0f}%", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, fontweight="bold", color=COR_TEXTO_FRACO,
            zorder=5)

    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25, zorder=0)
    if not mostrar_x:
        ax.set_xticklabels([])
    if not mostrar_y:
        ax.set_yticklabels([])


def gerar(simples: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    dias = sorted(simples["dia"].unique())
    w = np.sort(simples["banda_nm"].unique()).astype(float)

    n_linhas = len(PARES) * len(CONDICOES)
    fig, axes = plt.subplots(
        n_linhas, len(dias), figsize=(26, 17),
        gridspec_kw=dict(hspace=0.22, wspace=0.10,
                         top=0.905, bottom=0.075, left=0.075, right=0.985),
    )

    indice = simples.set_index(
        ["genotipo_a", "genotipo_b", "condicao", "dia"]
    ).sort_index()

    for i, (par, condicao) in enumerate(
        [(p, c) for p in PARES for c in CONDICOES]
    ):
        for j, dia in enumerate(dias):
            ax = axes[i, j]
            sub = indice.loc[(par[0], par[1], condicao, dia)]
            sub = sub.sort_values("banda_nm")
            desenhar_celula(
                ax, sub["banda_nm"].to_numpy(dtype=float),
                sub["q_fdr"].to_numpy(),
                mostrar_x=(i == n_linhas - 1), mostrar_y=(j == 0),
            )

            if i == 0:
                ax.set_title(dia, fontsize=11, fontweight="bold", pad=8)

        # Rotulo da linha: o par em cores dos genotipos, a condicao na cor dela.
        eixo = axes[i, 0]
        eixo.text(-0.44, 0.62, f"{par[0]}\nvs\n{par[1]}",
                  transform=eixo.transAxes, ha="center", va="center",
                  fontsize=8.5, fontweight="bold", color=COR_GENOTIPO[par[0]])
        eixo.text(-0.44, 0.16, condicao, transform=eixo.transAxes,
                  ha="center", va="center", fontsize=9, fontweight="bold",
                  color=COR_CONDICAO[condicao])

    for ax in axes[-1, :]:
        ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("-log10(q)", fontsize=8)

    # O teste vem do proprio CSV, nao fixado no titulo: a coluna `q_fdr` segue
    # `TESTE_EFEITO_SIMPLES`, e um titulo escrito a mao ficaria mentindo assim
    # que a constante mudasse.
    nome_teste = NOME_TESTE.get(
        simples["teste"].iloc[0], simples["teste"].iloc[0]
    )
    fig.suptitle(
        "Efeitos simples banda a banda: os dois genotipos do par diferem, "
        "dentro de cada condicao?\n"
        "42 combinacoes (3 pares x 2 condicoes x 7 dias) x 2051 bandas  -  "
        f"{nome_teste} com FDR de Benjamini-Hochberg, turno da manha",
        fontsize=15, fontweight="bold", y=0.975,
    )

    fig.legend(
        handles=[
            Line2D([0], [0], color=COR_CURVA, linewidth=1.2,
                   label="-log10 do q-valor por banda"),
            Line2D([0], [0], color=COR_LIMIAR, linestyle="--", linewidth=1.2,
                   label="Limiares q = 0.05, 0.01 e 0.001"),
        ] + [
            Patch(facecolor=TOM_ALPHA[a], label=f"Bandas com q < {a:g}")
            for a in ALPHAS
        ],
        loc="lower center", ncol=5, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.022),
    )

    fig.text(
        0.075, 0.004,
        "O numero no canto de cada celula e a porcentagem de bandas com "
        "q < 0.05. Faixas sombreadas mais escuras sobrevivem a limiares mais "
        "rigorosos.",
        fontsize=9, color=COR_TEXTO_FRACO, ha="left", va="bottom",
    )

    fig.savefig(SAIDA, dpi=200)
    print(f"Grade salva em: {SAIDA}")


def main() -> None:
    print("Gerando a grade dos efeitos simples...\n")
    simples = pd.read_csv(SAIDA_DIR / "efeitos_simples.csv", sep=";")
    print(f"  efeitos_simples: {len(simples)} linhas, "
          f"{simples.groupby(['dia', 'genotipo_a', 'genotipo_b', 'condicao']).ngroups}"
          f" combinacoes\n")
    gerar(simples)
    print("Concluido.")


if __name__ == "__main__":
    main()
