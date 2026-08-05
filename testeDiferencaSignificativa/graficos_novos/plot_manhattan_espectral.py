#!/usr/bin/env python3
"""Manhattan espectral: -log10(q FDR) ao longo do espectro, dia a dia.

O formato de GWAS e metabolomica aplicado ao eixo espectral. Cada linha do
grafico percorre as 2051 bandas; tudo acima da linha horizontal em
-log10(0,05) = 1,30 sobreviveu a correcao de Benjamini-Hochberg.

Duas colunas, que respondem perguntas diferentes:

Esquerda -- efeito do estresse hidrico dentro de cada genotipo
    Uma curva por material (`comparacao_estresse.csv`). Mostra em que bandas o
    deficit hidrico se manifesta e, sobretudo, que os tres materiais nao se
    manifestam nas mesmas bandas nem nos mesmos dias.

Direita -- termo de interacao genotipo x condicao (ART)
    Uma curva so (`interacao_por_banda.csv`). Onde ela passa da linha, o efeito
    do estresse depende do material -- e agrupar os genotipos num mesmo n
    estaria assumindo o contrario.

Sobre ler p-valores nesta figura
--------------------------------
Com 32 leituras por celula mas apenas 4 blocos independentes, o n efetivo esta
inflado e os q-valores estao otimistas: a altura absoluta das curvas nao deve
ser usada como medida de importancia. O que a figura entrega bem e a *forma* --
quais regioes espectrais sobem, quais dias sobem, e o contraste entre materiais
sob a mesma correcao. Para magnitude, use o delta de Cliff em
`comparacao_estresse.csv`.

O platô no topo de algumas curvas (CD202 em D09 e D10, BR16 em D05 e D06) nao e
um artefato de escala: com 32 leituras de cada lado, a separacao completa entre
os dois grupos ja produz o menor p que o teste de posto consegue emitir. Bandas
no plato estao no piso do teste, e a distancia entre elas nao e interpretavel --
mais uma razao para ordenar por delta de Cliff.

As colunas tem escalas de y diferentes (o termo de interacao chega mais alto
que o contraste de estresse). Sao dois graficos lado a lado, nunca dois eixos
no mesmo grafico -- dentro de cada coluna a escala e compartilhada pelos sete
dias, que e a comparacao que a figura convida a fazer.

Saida:
    manhattan_espectral.png

Uso:
    python plot_manhattan_espectral.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
DADOS = ROOT.parent / "dataset_gerado"
SAIDA = ROOT / "manhattan_espectral.png"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]

ALPHA = 0.05
LIMIAR = -np.log10(ALPHA)

# Paleta herdada de `plot_diferencas.py`, validada contra deuteranopia,
# protanopia e tritanopia.
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}
COR_INTERACAO = "#7b3fb5"
COR_LIMIAR = "#c2410c"
COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"
COR_GRADE = "#d9d8d4"

REGIOES = [
    ("VIS", 400, 700),
    ("red edge", 700, 780),
    ("NIR", 780, 1350),
    ("SWIR1", 1350, 1800),
    ("SWIR2", 1800, 2451),
]


def menos_log10(q: np.ndarray) -> np.ndarray:
    """-log10(q), protegido contra underflow para zero."""
    return -np.log10(np.clip(np.asarray(q, dtype=float), 1e-300, 1.0))


def moldura(ax: plt.Axes, w0: float, w1: float, ymax: float) -> None:
    """Grade, faixas de regiao e linha de limiar -- comum as duas colunas."""
    for regiao, ini, fim in REGIOES[1::2]:
        ax.axvspan(ini, min(fim, w1), color="#f7f6f3", zorder=0)

    ax.axhline(LIMIAR, color=COR_LIMIAR, linewidth=1.3, zorder=4)

    ax.set_xlim(w0, w1)
    ax.set_ylim(0, ymax)
    ax.grid(True, color=COR_GRADE, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=COR_TEXTO_FRACO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def painel_estresse(ax: plt.Axes, sub: pd.DataFrame, ymax: float) -> None:
    """Uma curva por genotipo: o contraste IRRIG vs NIRRIG daquele dia."""
    w0, w1 = sub["banda_nm"].min(), sub["banda_nm"].max()
    moldura(ax, w0, w1, ymax)

    for genotipo in GENOTIPOS:
        linha = sub[sub["genotipo"] == genotipo].sort_values("banda_nm")
        ax.plot(linha["banda_nm"], menos_log10(linha["q_fdr"]),
                color=COR_GENOTIPO[genotipo], linewidth=1.1, zorder=3)


def painel_interacao(ax: plt.Axes, sub: pd.DataFrame, ymax: float) -> None:
    """Curva unica: o termo de interacao genotipo x condicao do ART."""
    linha = sub.sort_values("banda_nm")
    w0, w1 = linha["banda_nm"].min(), linha["banda_nm"].max()
    moldura(ax, w0, w1, ymax)

    y = menos_log10(linha["q_interacao"])
    ax.fill_between(linha["banda_nm"], LIMIAR, y, where=(y > LIMIAR),
                    color=COR_INTERACAO, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(linha["banda_nm"], y, color=COR_INTERACAO, linewidth=1.2, zorder=3)

    prop = float((linha["q_interacao"] < ALPHA).mean())
    ax.text(0.985, 0.9, f"{prop:.0%} das bandas com interacao",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=COR_TEXTO_FRACO,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.85))


def rotular_regioes(ax: plt.Axes) -> None:
    """Nomes das regioes espectrais abaixo dos tiques do eixo x."""
    for regiao, ini, fim in REGIOES:
        if regiao == "red edge":
            continue
        ax.text((ini + min(fim, 2450)) / 2, -0.20, regiao,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5,
                color=COR_TEXTO_FRACO, style="italic")


def gerar(estresse: pd.DataFrame, interacao: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # Escala propria por coluna, compartilhada pelos sete dias.
    ymax_estresse = float(np.ceil(menos_log10(estresse["q_fdr"]).max())) + 0.5
    ymax_interacao = float(np.ceil(menos_log10(interacao["q_interacao"]).max())) + 0.5

    fig = plt.figure(figsize=(16, 19))
    gs = fig.add_gridspec(
        len(DIAS), 2, width_ratios=[1.55, 1],
        hspace=0.34, wspace=0.13,
        top=0.912, bottom=0.055, left=0.06, right=0.985,
    )

    for i, dia in enumerate(DIAS):
        ax_e = fig.add_subplot(gs[i, 0])
        ax_i = fig.add_subplot(gs[i, 1])

        painel_estresse(ax_e, estresse[estresse["dia"] == dia], ymax_estresse)
        painel_interacao(ax_i, interacao[interacao["dia"] == dia], ymax_interacao)

        ax_e.set_ylabel(f"{dia}\n-log10(q FDR)", fontsize=9.5, color=COR_TEXTO)

        if i == 0:
            ax_e.set_title("Efeito do estresse dentro de cada genotipo",
                           fontsize=12.5, fontweight="bold", color=COR_TEXTO,
                           pad=10)
            ax_i.set_title("Interacao genotipo x condicao (ART)",
                           fontsize=12.5, fontweight="bold", color=COR_TEXTO,
                           pad=10)

        if i == len(DIAS) - 1:
            for ax in (ax_e, ax_i):
                ax.set_xlabel("Comprimento de onda (nm)", fontsize=10.5,
                              color=COR_TEXTO, labelpad=44)
                rotular_regioes(ax)
        else:
            ax_e.set_xticklabels([])
            ax_i.set_xticklabels([])

    legenda = (
        [Line2D([0], [0], color=COR_GENOTIPO[g], linewidth=2.2,
                label=f"Estresse em {g} (IRRIG vs NIRRIG)")
         for g in GENOTIPOS]
        + [Line2D([0], [0], color=COR_INTERACAO, linewidth=2.2,
                  label="Interacao genotipo x condicao"),
           Line2D([0], [0], color=COR_LIMIAR, linewidth=2.2,
                  label=f"Limiar q = {ALPHA} (-log10 = {LIMIAR:.2f})".replace(".", ","))]
    )
    fig.legend(handles=legenda, loc="upper center", ncol=5, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.948), labelcolor=COR_TEXTO)

    fig.suptitle(
        "Significancia ao longo do espectro, dia a dia\n"
        "-log10(q FDR) sobre 2051 bandas, turno da manha -- acima da linha "
        "laranja a banda sobreviveu ao FDR de Benjamini-Hochberg",
        fontsize=15, fontweight="bold", y=0.978, color=COR_TEXTO,
    )

    fig.savefig(SAIDA, dpi=300)
    print(f"Figura salva em: {SAIDA}")


def main() -> None:
    print("Gerando Manhattan espectral...\n")

    estresse = pd.read_csv(DADOS / "comparacao_estresse.csv", sep=";")
    interacao = pd.read_csv(DADOS / "interacao_por_banda.csv", sep=";")

    print(f"  comparacao_estresse: {len(estresse)} linhas")
    print(f"  interacao_por_banda: {len(interacao)} linhas\n")

    gerar(estresse, interacao)
    print("Concluido.")


if __name__ == "__main__":
    main()
