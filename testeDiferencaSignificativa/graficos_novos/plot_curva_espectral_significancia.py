#!/usr/bin/env python3
"""Curva espectral media com faixa de bandas significativas (grade dia x genotipo).

O formato classico de espectroscopia: a curva de refletancia de cada grupo e,
logo abaixo, uma faixa marcando onde as duas curvas diferem. Aqui o contraste e
o do estresse hidrico -- IRRIG contra NIRRIG -- rodado DENTRO de cada genotipo,
que e exatamente a coluna que `comparacao_genotipo_condicao.py` isola em
`comparacao_estresse.csv` com familia de FDR propria.

A grade e 7 dias x 3 genotipos porque essa e a estrutura do resultado: o efeito
do estresse nao e constante nem entre materiais nem ao longo da campanha. Uma
figura unica com as curvas agregadas esconderia justamente isso.

Duas escalas convivem na figura, e a distincao importa
-------------------------------------------------------
- A curva mostra refletancia do estagio `suavizado` (recorte + jump correction
  + Savitzky-Golay), em %. E a grandeza fisica que o leitor espera ver.
- A faixa de significancia vem dos testes, que rodaram sobre o estagio
  `normalizado` (SNV). O SNV remove a magnitude de cada leitura e preserva a
  forma, entao ele testa forma espectral, nao brilho.

A consequencia pratica: existem bandas marcadas como significativas em que as
duas curvas de refletancia quase se tocam. Nao e contradicao -- e o SNV fazendo
o que foi pedido dele, separar por forma depois de remover o efeito de
magnitude (angulo de folha, geometria de iluminacao) que domina a refletancia
bruta e nao e sinal biologico.

A cor da faixa carrega a direcao do efeito, lida pelo sinal do delta de Cliff:
laranja quando o nao-irrigado refletiu mais, azul quando o irrigado refletiu
mais. Cinza claro e banda nao significativa.

Paleta herdada de `plot_diferencas.py`, ja validada contra deuteranopia,
protanopia e tritanopia.

Saida:
    curva_espectral_significancia.png

Uso:
    python plot_curva_espectral_significancia.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
MODULO = ROOT.parent
sys.path.insert(0, str(MODULO.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(MODULO.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402

DADOS = MODULO / "dataset_gerado"
SAIDA = ROOT / "curva_espectral_significancia.png"

TURNO = "manha"
# A curva mostra refletancia; os testes rodaram sobre o SNV. Ver docstring.
ESTAGIO_CURVA = "suavizado"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]

COR_CONDICAO = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}
COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"
COR_NAO_SIG = "#e3e2df"
COR_GRADE = "#d9d8d4"

REGIOES = [
    ("VIS", 400, 700),
    ("red edge", 700, 780),
    ("NIR", 780, 1350),
    ("SWIR1", 1350, 1800),
    ("SWIR2", 1800, 2451),
]


def curvas_por_celula(
    meta: pd.DataFrame,
    espectro: np.ndarray,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Media e desvio padrao da refletancia em cada dia x genotipo x condicao."""
    curvas = {}
    for chave, idx in meta.groupby(["dia", "genotipo", "condicao"]).indices.items():
        bloco = espectro[idx]
        curvas[chave] = (np.nanmean(bloco, axis=0), np.nanstd(bloco, axis=0))
    return curvas


def faixa_significancia(ax: plt.Axes, w: np.ndarray, sub: pd.DataFrame) -> None:
    """Desenha a barra de bandas significativas, colorida pela direcao do efeito.

    `imshow` em vez de um `axvspan` por banda: sao 2051 bandas por celula e 21
    celulas, e desenhar 43 mil patches deixa o render lento sem ganho visual.
    """
    sig = sub["significativa"].to_numpy(dtype=bool)
    delta = sub["delta_cliff"].to_numpy(dtype=float)

    # 0 = nao significativa, 1 = IRRIG acima, 2 = NIRRIG acima.
    codigo = np.zeros(len(sig), dtype=int)
    codigo[sig & (delta < 0)] = 1
    codigo[sig & (delta >= 0)] = 2

    cores = [COR_NAO_SIG, COR_CONDICAO["IRRIG"], COR_CONDICAO["NIRRIG"]]

    ax.imshow(
        codigo[np.newaxis, :],
        cmap=ListedColormap(cores),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
        aspect="auto", interpolation="nearest",
        extent=(w[0], w[-1], 0, 1),
    )
    ax.set_yticks([])
    ax.set_xlim(w[0], w[-1])
    for lado in ax.spines.values():
        lado.set_visible(False)


def celula(
    ax_curva: plt.Axes,
    ax_faixa: plt.Axes,
    w: np.ndarray,
    curvas: dict,
    estresse: pd.DataFrame,
    dia: str,
    genotipo: str,
    primeira_linha: bool,
    primeira_coluna: bool,
    ultima_linha: bool,
) -> None:
    """Uma celula da grade: curvas IRRIG/NIRRIG e a faixa de significancia."""
    for regiao, ini, fim in REGIOES[1::2]:
        ax_curva.axvspan(ini, fim, color="#f7f6f3", zorder=0)

    for condicao in ("IRRIG", "NIRRIG"):
        media, desvio = curvas[(dia, genotipo, condicao)]
        cor = COR_CONDICAO[condicao]
        ax_curva.fill_between(
            w, (media - desvio) * 100, (media + desvio) * 100,
            color=cor, alpha=0.13, linewidth=0, zorder=2,
        )
        ax_curva.plot(w, media * 100, color=cor, linewidth=1.6, zorder=3)

    sub = estresse[(estresse["dia"] == dia) & (estresse["genotipo"] == genotipo)]
    sub = sub.sort_values("banda_nm")
    faixa_significancia(ax_faixa, w, sub)

    prop = float(sub["significativa"].mean())
    ax_curva.text(
        0.985, 0.04, f"{prop:.0%} das bandas",
        transform=ax_curva.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=COR_TEXTO_FRACO,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor="none", alpha=0.82),
    )

    ax_curva.set_xlim(w[0], w[-1])
    ax_curva.set_ylim(0, 58)
    ax_curva.set_xticklabels([])
    ax_curva.tick_params(labelsize=8, colors=COR_TEXTO_FRACO)
    ax_curva.grid(True, color=COR_GRADE, linewidth=0.5, alpha=0.7)
    ax_curva.set_axisbelow(True)
    for lado in ("top", "right"):
        ax_curva.spines[lado].set_visible(False)

    if primeira_linha:
        ax_curva.set_title(genotipo, fontsize=13, fontweight="bold",
                           color=COR_TEXTO, pad=10)
    if primeira_coluna:
        ax_curva.set_ylabel(f"{dia}\nRefletancia (%)", fontsize=9.5,
                            color=COR_TEXTO)
    else:
        ax_curva.set_yticklabels([])

    if ultima_linha:
        ax_faixa.set_xlabel("Comprimento de onda (nm)", fontsize=10,
                            color=COR_TEXTO)
        ax_faixa.tick_params(labelsize=8.5, colors=COR_TEXTO_FRACO)
    else:
        ax_faixa.set_xticklabels([])
        ax_faixa.tick_params(labelsize=0, length=0)


def marcar_regioes(ax: plt.Axes) -> None:
    """Rotula as regioes espectrais no topo da grade."""
    for regiao, ini, fim in REGIOES:
        if regiao == "red edge":
            continue
        ax.text((ini + min(fim, 2450)) / 2, 55.5, regiao,
                ha="center", va="top", fontsize=8, color=COR_TEXTO_FRACO,
                style="italic")


def gerar(meta: pd.DataFrame, espectro: np.ndarray, w: np.ndarray,
          estresse: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    curvas = curvas_por_celula(meta, espectro)

    fig = plt.figure(figsize=(16.5, 20))
    gs = fig.add_gridspec(
        len(DIAS), len(GENOTIPOS),
        hspace=0.30, wspace=0.09,
        top=0.925, bottom=0.055, left=0.075, right=0.975,
    )

    for i, dia in enumerate(DIAS):
        for j, genotipo in enumerate(GENOTIPOS):
            sub_gs = gs[i, j].subgridspec(2, 1, height_ratios=[6, 0.85],
                                          hspace=0.05)
            ax_curva = fig.add_subplot(sub_gs[0])
            ax_faixa = fig.add_subplot(sub_gs[1])
            celula(ax_curva, ax_faixa, w, curvas, estresse, dia, genotipo,
                   primeira_linha=(i == 0), primeira_coluna=(j == 0),
                   ultima_linha=(i == len(DIAS) - 1))
            if i == 0:
                marcar_regioes(ax_curva)

    legenda = [
        Line2D([0], [0], color=COR_CONDICAO["IRRIG"], linewidth=2.2,
               label="IRRIG -- media +/- 1 desvio padrao"),
        Line2D([0], [0], color=COR_CONDICAO["NIRRIG"], linewidth=2.2,
               label="NIRRIG -- media +/- 1 desvio padrao"),
        Patch(facecolor=COR_CONDICAO["IRRIG"],
              label="Faixa: banda significativa, IRRIG refletiu mais"),
        Patch(facecolor=COR_CONDICAO["NIRRIG"],
              label="Faixa: banda significativa, NIRRIG refletiu mais"),
        Patch(facecolor=COR_NAO_SIG, label="Faixa: nao significativa"),
    ]
    fig.legend(handles=legenda, loc="upper center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.952),
               labelcolor=COR_TEXTO, columnspacing=1.6, handlelength=1.6)

    fig.suptitle(
        "Efeito do estresse hidrico no espectro, dentro de cada genotipo\n"
        "Curva: refletancia media (Savitzky-Golay). Faixa inferior: bandas com "
        "q FDR < 0,05 no Kruskal-Wallis IRRIG vs NIRRIG sobre o espectro SNV",
        fontsize=15, fontweight="bold", y=0.985, color=COR_TEXTO,
    )

    fig.savefig(SAIDA, dpi=300)
    print(f"Figura salva em: {SAIDA}")


def main() -> None:
    print("Carregando espectro pre-processado...")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    espectro = estagios[ESTAGIO_CURVA]
    print(f"  {len(meta)} leituras do turno '{TURNO}' x {len(w)} bandas")

    estresse = pd.read_csv(DADOS / "comparacao_estresse.csv", sep=";")
    print(f"  comparacao_estresse: {len(estresse)} linhas\n")

    gerar(meta, espectro, w, estresse)
    print("Concluido.")


if __name__ == "__main__":
    main()
