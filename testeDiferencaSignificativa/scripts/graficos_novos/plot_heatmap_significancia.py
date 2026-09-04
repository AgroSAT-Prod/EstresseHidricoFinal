#!/usr/bin/env python3
"""Heatmap de significancia: dia x comprimento de onda, para cada contraste.

Resume os sete dias e as 2051 bandas numa figura so. Cada bloco e um contraste
diferente do delineamento, todos na mesma escala de cor e no mesmo eixo x, o
que permite ler as tres perguntas do modulo em sequencia vertical:

A) Omnibus das 6 celulas -- ha qualquer diferenca? (`comparacao_omnibus.csv`)
B) Efeito do estresse dentro de cada genotipo, um bloco por material
   (`comparacao_estresse.csv`)
C) Termo de interacao genotipo x condicao por ART (`interacao_por_banda.csv`)

A leitura que a figura entrega de graca: o bloco A satura -- quase tudo e
significativo, o que e efeito do n inflado por pseudorreplicacao e nao
informacao. Os blocos B mostram que o estresse aparece em momentos diferentes
conforme o material. E o bloco C mostra que a interacao nao e residual: ela
cobre o espectro inteiro em cinco dos sete dias, que e o argumento contra
agrupar os genotipos num mesmo n.

Cor
---
-log10(q FDR), rampa sequencial de um unico tom (claro = nao significativo,
escuro = muito significativo). A barra de cor traz o limiar q = 0,05 marcado
como tique, entao a fronteira entre "nao significativo" e "significativo" e
lida na propria legenda em vez de exigir uma segunda figura binaria.

A escala satura em VMAX. O omnibus chega a -log10(q) = 28,8, e deixar a escala
livre comprimiria todos os outros blocos contra o branco. O corte mantem os
blocos comparaveis entre si -- e acima de 6 a distincao ja nao muda nenhuma
decisao.

Saida:
    heatmap_significancia.png

Uso:
    python plot_heatmap_significancia.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
DADOS = ROOT.parent.parent / "resultados" / "dataset_gerado"
SAIDA = ROOT.parent.parent / "resultados" / "heatmap_significancia.png"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]

ALPHA = 0.05
LIMIAR = -np.log10(ALPHA)  # 1.301
VMAX = 6.0

COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"

# Rampa sequencial de tom unico, do branco-papel ao azul-escuro. Mesma familia
# do CMAP_SEQUENCIAL de `plot_diferencas.py`.
CMAP = LinearSegmentedColormap.from_list(
    "significancia", ["#f7f9fc", "#c8d9ee", "#7fa8d9", "#3a6ba5", "#12294d"]
)

REGIOES = [
    ("VIS", 400, 700),
    ("red edge", 700, 780),
    ("NIR", 780, 1350),
    ("SWIR1", 1350, 1800),
    ("SWIR2", 1800, 2451),
]


def matriz(df: pd.DataFrame, coluna_q: str) -> tuple[np.ndarray, np.ndarray]:
    """Monta a matriz dia x banda de -log10(q) e o vetor de comprimentos."""
    largo = df.pivot_table(index="dia", columns="banda_nm", values=coluna_q)
    largo = largo.reindex(DIAS)
    w = largo.columns.to_numpy(dtype=float)
    # q vem sempre > 0 nestes CSVs; o clip protege contra underflow futuro.
    return -np.log10(np.clip(largo.to_numpy(dtype=float), 1e-300, 1.0)), w


def bloco(
    ax: plt.Axes,
    dados: np.ndarray,
    w: np.ndarray,
    titulo: str,
    subtitulo: str,
    ultimo: bool,
) -> None:
    """Desenha um heatmap dia x banda."""
    ax.imshow(
        dados, cmap=CMAP, vmin=0, vmax=VMAX,
        aspect="auto", interpolation="nearest",
        extent=(w[0], w[-1], len(DIAS) - 0.5, -0.5),
    )

    ax.set_yticks(range(len(DIAS)))
    ax.set_yticklabels(DIAS, fontsize=9.5, color=COR_TEXTO)
    ax.set_xlim(w[0], w[-1])
    ax.tick_params(length=0, colors=COR_TEXTO_FRACO)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)

    # Separadores das regioes espectrais, por cima do heatmap.
    for _, ini, _ in REGIOES[1:]:
        ax.axvline(ini, color="white", linewidth=0.9, alpha=0.55)

    ax.set_title(titulo, fontsize=12, fontweight="bold", color=COR_TEXTO,
                 loc="left", pad=24)
    ax.text(0.0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9.5,
            color=COR_TEXTO_FRACO, ha="left", va="bottom")

    if ultimo:
        ax.set_xlabel("Comprimento de onda (nm)", fontsize=11, color=COR_TEXTO,
                      labelpad=30)
        ax.tick_params(labelsize=9.5)
        rotular_regioes(ax)
    else:
        ax.set_xticklabels([])


def rotular_regioes(ax: plt.Axes) -> None:
    """Nomes das regioes espectrais entre os tiques e o rotulo do eixo x.

    As linhas brancas verticais ja separam as regioes em todos os blocos; estes
    rotulos aparecem uma unica vez, no bloco de baixo, para nao competirem com
    os titulos dos blocos.
    """
    for regiao, ini, fim in REGIOES:
        if regiao == "red edge":
            continue
        ax.text((ini + min(fim, 2450)) / 2, -0.155, regiao,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9,
                color=COR_TEXTO_FRACO, style="italic")


def gerar(omnibus: pd.DataFrame, estresse: pd.DataFrame,
          interacao: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    blocos = [
        (*matriz(omnibus, "q_fdr"),
         "A. Omnibus das 6 celulas (3 genotipos x 2 condicoes)",
         "Kruskal-Wallis -- responde apenas se ha alguma diferenca"),
    ]
    for genotipo in GENOTIPOS:
        sub = estresse[estresse["genotipo"] == genotipo]
        # Faixa de bandas significativas ao longo dos dias: e o que separa um
        # material que responde sempre de um que responde so em alguns dias.
        por_dia = sub.groupby("dia")["significativa"].mean()
        faixa = (f"Kruskal-Wallis IRRIG vs NIRRIG, familia de FDR propria -- "
                 f"de {por_dia.min():.0%} ({por_dia.idxmin()}) a "
                 f"{por_dia.max():.0%} ({por_dia.idxmax()}) das bandas")
        blocos.append((
            *matriz(sub, "q_fdr"),
            f"B. Efeito do estresse dentro de {genotipo}",
            faixa,
        ))
    blocos.append((
        *matriz(interacao, "q_interacao"),
        "C. Interacao genotipo x condicao (ART)",
        "onde ha cor, a resposta ao estresse depende do material",
    ))

    fig = plt.figure(figsize=(15, 16.5))
    gs = fig.add_gridspec(
        len(blocos), 2, width_ratios=[60, 1],
        hspace=0.46, wspace=0.03,
        top=0.888, bottom=0.055, left=0.065, right=0.93,
    )

    imagem = None
    for i, (dados, w, titulo, subtitulo) in enumerate(blocos):
        ax = fig.add_subplot(gs[i, 0])
        bloco(ax, dados, w, titulo, subtitulo, ultimo=(i == len(blocos) - 1))
        if i == 0:
            imagem = ax.images[0]

    cax = fig.add_subplot(gs[1:4, 1])
    barra = fig.colorbar(imagem, cax=cax, extend="max")
    barra.set_label("-log10(q FDR)", fontsize=11, color=COR_TEXTO)
    barra.set_ticks([0, LIMIAR, 2, 3, 4, 5, 6])
    barra.set_ticklabels(["0", f"{LIMIAR:.2f}\nq = {ALPHA}".replace(".", ","),
                          "2", "3", "4", "5", "6"])
    barra.ax.tick_params(labelsize=9, colors=COR_TEXTO_FRACO)
    barra.ax.axhline(LIMIAR, color="#ad1457", linewidth=1.6)
    barra.outline.set_visible(False)

    fig.suptitle(
        "Significancia banda a banda ao longo da campanha\n"
        "7 dias x 2051 bandas, turno da manha, FDR de Benjamini-Hochberg com as "
        "bandas como familia\n"
        "A linha vinho na barra de cor marca o limiar q = 0,05",
        fontsize=15, fontweight="bold", y=0.975, color=COR_TEXTO,
    )

    fig.savefig(SAIDA, dpi=300)
    print(f"Figura salva em: {SAIDA}")


def main() -> None:
    print("Gerando heatmap de significancia...\n")

    omnibus = pd.read_csv(DADOS / "comparacao_omnibus.csv", sep=";")
    estresse = pd.read_csv(DADOS / "comparacao_estresse.csv", sep=";")
    interacao = pd.read_csv(DADOS / "interacao_por_banda.csv", sep=";")

    print(f"  comparacao_omnibus: {len(omnibus)} linhas")
    print(f"  comparacao_estresse: {len(estresse)} linhas")
    print(f"  interacao_por_banda: {len(interacao)} linhas\n")

    gerar(omnibus, estresse, interacao)
    print("Concluido.")


if __name__ == "__main__":
    main()
