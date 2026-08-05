#!/usr/bin/env python3
"""Grade completa dos contrastes entre genotipos: 12 contrastes x 7 dias.

Mesma leitura de `plot_grade_efeitos_simples.py`, estendida com os contrastes
em que genotipo e condicao mudam ao mesmo tempo. Cada par de genotipos ocupa um
bloco de quatro linhas:

    A IRRIG  vs B IRRIG     <- mesma condicao (efeitos simples)
    A NIRRIG vs B NIRRIG    <- mesma condicao (efeitos simples)
    A IRRIG  vs B NIRRIG    <- cruzado
    A NIRRIG vs B IRRIG     <- cruzado

As duas primeiras linhas medem a distancia entre materiais com o estresse
controlado. As duas ultimas poem genotipo e estresse um contra o outro: se uma
linha cruzada mostra MENOS separacao que as de mesma condicao acima dela, os
dois efeitos se cancelam naquele dia -- e o material irrigado fica
espectralmente confundivel com o outro material sob deficit.

A assimetria entre as duas linhas cruzadas de um bloco e a interacao lida na
escala do contraste: se genotipo e condicao fossem aditivos, somar os dois
efeitos (uma linha) e subtrair (a outra) daria resultados espelhados.

Cada celula mostra, ao longo do espectro:

- a curva de -log10 do q-valor de Benjamini-Hochberg, banda a banda;
- tres linhas horizontais nos limiares 0.05, 0.01 e 0.001;
- as regioes significativas sombreadas em tres tons encaixados, um por limiar.
  Quanto mais escuro, mais rigoroso o criterio que a regiao sobrevive.

A leitura util e a **largura e a posicao** das faixas sombreadas, nao a altura
da curva: com o n deste experimento a curva satura no topo da escala em boa
parte do espectro (o maximo real de -log10(q) e 18.0, e o eixo vai a 10).

Fontes: `efeitos_simples.csv` para as linhas de mesma condicao e
`efeitos_cruzados.csv` para as cruzadas. Os dois saem do mesmo teste, com a
mesma familia de FDR, entao as 12 linhas sao comparaveis entre si.

Uso:
    python plot_grade_celulas_cruzadas.py
"""

from __future__ import annotations

import sys
from itertools import combinations
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
from efeitos_cruzados import CRUZAMENTOS  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"
SAIDA = ROOT / "grade_celulas_cruzadas.png"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]
PARES = list(combinations(GENOTIPOS, 2))

COR_CONDICAO = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}

# Tons encaixados, um por limiar: mais escuro = criterio mais rigoroso.
TOM_ALPHA = {0.05: "#dbe6f4", 0.01: "#a9c6e8", 0.001: "#6a9bd1"}

COR_CURVA = "#1a1a19"
COR_LIMIAR = "#c2410c"
COR_TEXTO_FRACO = "#52514e"
COR_DIVISOR = "#9a9894"

Y_MAX = 10.0

NOME_TESTE = {
    "mannwhitney": "Mann-Whitney exato",
    "kruskal": "Kruskal-Wallis",
}


def carregar_contrastes() -> pd.DataFrame:
    """Une efeitos simples e cruzados numa tabela so, com as celulas explicitas.

    Os efeitos simples nao trazem `condicao_a`/`condicao_b` porque neles as
    duas celulas estao, por definicao, na mesma condicao -- aqui a coluna e
    duplicada para que as 12 linhas da grade tenham o mesmo formato.
    """
    colunas = ["dia", "genotipo_a", "genotipo_b", "banda_nm", "q_fdr", "teste"]

    simples = pd.read_csv(
        SAIDA_DIR / "efeitos_simples.csv", sep=";",
        usecols=colunas + ["condicao"],
    )
    simples["condicao_a"] = simples["condicao"]
    simples["condicao_b"] = simples["condicao"]
    simples["tipo"] = "mesma condicao"
    simples = simples.drop(columns="condicao")

    cruzados = pd.read_csv(
        SAIDA_DIR / "efeitos_cruzados.csv", sep=";",
        usecols=colunas + ["condicao_a", "condicao_b"],
    )
    cruzados["tipo"] = "cruzado"

    return pd.concat([simples, cruzados], ignore_index=True)


def linhas_da_grade() -> list[tuple[str, str, str, str, str]]:
    """As 12 linhas, agrupadas em blocos de quatro por par de genotipos."""
    linhas = []
    for a, b in PARES:
        for condicao in CONDICOES:
            linhas.append((a, condicao, b, condicao, "mesma condicao"))
        for cond_a, cond_b in CRUZAMENTOS:
            linhas.append((a, cond_a, b, cond_b, "cruzado"))
    return linhas


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


def rotular_linha(
    ax: plt.Axes,
    gen_a: str,
    cond_a: str,
    gen_b: str,
    cond_b: str,
    tipo: str,
) -> None:
    """Rotulo da margem esquerda: as duas celulas empilhadas, uma sobre a outra.

    O genotipo vai na cor do material e a condicao na cor dela, entao o tipo do
    contraste se le de relance: duas condicoes da mesma cor e uma comparacao
    dentro da condicao, duas cores diferentes e um cruzamento.
    """
    posicoes = [
        (0.93, gen_a, COR_GENOTIPO[gen_a], 8.5, "bold"),
        (0.75, cond_a, COR_CONDICAO[cond_a], 7.5, "bold"),
        (0.52, "vs", COR_TEXTO_FRACO, 7.0, "normal"),
        (0.29, gen_b, COR_GENOTIPO[gen_b], 8.5, "bold"),
        (0.11, cond_b, COR_CONDICAO[cond_b], 7.5, "bold"),
    ]
    for y, texto, cor, tamanho, peso in posicoes:
        ax.text(-0.44, y, texto, transform=ax.transAxes, ha="center",
                va="center", fontsize=tamanho, fontweight=peso, color=cor)

    if tipo == "cruzado":
        ax.text(-0.60, 0.52, "cruzado", transform=ax.transAxes, ha="center",
                va="center", fontsize=7, color=COR_TEXTO_FRACO, rotation=90)


def gerar(contrastes: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    dias = sorted(contrastes["dia"].unique())
    w = np.sort(contrastes["banda_nm"].unique()).astype(float)
    linhas = linhas_da_grade()

    fig, axes = plt.subplots(
        len(linhas), len(dias), figsize=(26, 31),
        gridspec_kw=dict(hspace=0.22, wspace=0.10,
                         top=0.948, bottom=0.041, left=0.075, right=0.985),
    )

    indice = contrastes.set_index(
        ["genotipo_a", "condicao_a", "genotipo_b", "condicao_b", "dia"]
    ).sort_index()

    for i, (gen_a, cond_a, gen_b, cond_b, tipo) in enumerate(linhas):
        for j, dia in enumerate(dias):
            ax = axes[i, j]
            sub = indice.loc[(gen_a, cond_a, gen_b, cond_b, dia)]
            sub = sub.sort_values("banda_nm")
            desenhar_celula(
                ax, sub["banda_nm"].to_numpy(dtype=float),
                sub["q_fdr"].to_numpy(),
                mostrar_x=(i == len(linhas) - 1), mostrar_y=(j == 0),
            )

            if i == 0:
                ax.set_title(dia, fontsize=11, fontweight="bold", pad=8)

        rotular_linha(axes[i, 0], gen_a, cond_a, gen_b, cond_b, tipo)

    for ax in axes[-1, :]:
        ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("-log10(q)", fontsize=8)

    # Divisor entre os blocos de quatro linhas, para o olho nao misturar pares.
    for corte in range(len(CONDICOES) + len(CRUZAMENTOS), len(linhas),
                       len(CONDICOES) + len(CRUZAMENTOS)):
        y = (axes[corte - 1, 0].get_position().y0
             + axes[corte, 0].get_position().y1) / 2
        fig.add_artist(Line2D([0.035, 0.99], [y, y], color=COR_DIVISOR,
                              linewidth=1.0, alpha=0.7))

    nome_teste = NOME_TESTE.get(
        contrastes["teste"].iloc[0], contrastes["teste"].iloc[0]
    )
    fig.suptitle(
        "Contrastes entre genotipos, banda a banda: dentro da mesma condicao "
        "e com as condicoes cruzadas\n"
        "84 combinacoes (3 pares x 4 contrastes x 7 dias) x 2051 bandas  -  "
        f"{nome_teste} com FDR de Benjamini-Hochberg, turno da manha",
        fontsize=15, fontweight="bold", y=0.985,
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
        bbox_to_anchor=(0.5, 0.012),
    )

    fig.text(
        0.075, 0.002,
        "O numero no canto de cada celula e a porcentagem de bandas com "
        "q < 0.05. Nas duas linhas cruzadas de cada bloco, genotipo e condicao "
        "mudam juntos: menos faixa sombreada que nas linhas de mesma condicao "
        "significa que os dois efeitos se cancelam.",
        fontsize=9, color=COR_TEXTO_FRACO, ha="left", va="bottom",
    )

    fig.savefig(SAIDA, dpi=160)
    print(f"Grade salva em: {SAIDA}")


def main() -> None:
    print("Gerando a grade dos contrastes entre genotipos...\n")
    contrastes = carregar_contrastes()
    combos = contrastes.groupby(
        ["dia", "genotipo_a", "condicao_a", "genotipo_b", "condicao_b"]
    ).ngroups
    print(f"  {len(contrastes)} linhas, {combos} combinacoes")
    for tipo, sub in contrastes.groupby("tipo"):
        print(f"    {tipo:<16} {sub['q_fdr'].lt(ALPHAS[0]).mean():.1%} "
              f"das bandas com q < {ALPHAS[0]:g}")
    print()
    gerar(contrastes)
    print("Concluido.")


if __name__ == "__main__":
    main()
