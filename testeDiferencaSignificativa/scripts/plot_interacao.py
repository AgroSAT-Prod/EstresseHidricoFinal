#!/usr/bin/env python3
"""Painel da interacao genotipo x condicao em tres niveis de significancia.

A pergunta e se os genotipos podem ser agrupados. O veredito depende de onde
se corta a significancia, entao a figura mostra os tres niveis lado a lado --
0.05, 0.01 e 0.001 sobre o q-valor de Benjamini-Hochberg. O que nao muda entre
eles e conclusao; o que muda era artefato do limiar.

A) ART no modelo completo (3 genotipos x 2 condicoes), um painel por nivel.
B) Interacao par a par, uma matriz por nivel. Localiza qual genotipo carrega
   a interacao.
C) Efeitos simples dentro de IRRIG e de NIRRIG, no nivel de referencia.
D) Veredito por par e dia, uma matriz por nivel.
E) Estabilidade: em quantos dos tres niveis o veredito coincide.

Um alerta de leitura embutido no painel D: apertar o alfa reduz o numero de
bandas significativas, e como o veredito premia a NAO significancia, niveis
mais rigorosos empurram mecanicamente para "agrupavel". Isso e perda de poder,
nao evidencia de equivalencia -- ver a nota no rodape da figura.

Uso:
    python plot_interacao.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from interacao_genotipo_condicao import (  # noqa: E402
    ALPHA_REFERENCIA,
    ALPHAS,
    LIMIAR_AGRUPAR,
    SIGLA_VEREDITO,
    coluna_sig,
)

SAIDA_DIR = ROOT.parent / "resultados" / "dataset_gerado"
SAIDA = ROOT.parent / "resultados" / "interacao_painel.png"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]

COR_TERMO = {
    "interacao": "#ad1457",
    "condicao": "#eb6834",
    "genotipo": "#1baf7a",
}
ROTULO_TERMO = {
    "interacao": "Interacao genotipo x condicao",
    "condicao": "Condicao (IRRIG vs NIRRIG)",
    "genotipo": "Genotipo (3 materiais)",
}

# Cores de status, validadas: pior dE 20.8 em visao normal e 8.1 sob CVD.
# Cada celula leva rotulo em texto -- a identidade nunca depende so da cor.
COR_VEREDITO = {
    "agrupavel": "#1baf7a",
    "offset constante": "#eda100",
    "nao agrupavel": "#e34948",
}
ORDEM_VEREDITO = ["agrupavel", "offset constante", "nao agrupavel"]

COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"

CMAP_SEQUENCIAL = LinearSegmentedColormap.from_list(
    "magnitude", ["#f4f6fa", "#7fa8d9", "#1a3a6b"]
)
CMAP_ESTABILIDADE = LinearSegmentedColormap.from_list(
    "estabilidade", ["#e34948", "#eda100", "#1baf7a"], N=3
)


def rotulo_alpha(alpha: float) -> str:
    return f"q < {alpha:g}"


def anotar_grade(ax, grade, fonte=8):
    """Escreve o valor em cada celula, com contraste contra o fundo."""
    for i in range(grade.shape[0]):
        for j in range(grade.shape[1]):
            valor = grade[i, j]
            if np.isnan(valor):
                continue
            ax.text(j, i, f"{valor:.0f}", ha="center", va="center",
                    fontsize=fonte, color="white" if valor > 55 else COR_TEXTO)


def grade_por_par(resumo: pd.DataFrame, coluna: str, dias: list[str],
                  pares: pd.DataFrame) -> np.ndarray:
    """Matriz pares x dias de uma coluna do resumo."""
    return np.array([
        resumo[(resumo["genotipo_a"] == a) & (resumo["genotipo_b"] == b)]
        .set_index("dia")[coluna].reindex(dias).to_numpy()
        for a, b in zip(pares["genotipo_a"], pares["genotipo_b"])
    ], dtype=float)


def painel_a(axes: list[plt.Axes], completo: pd.DataFrame) -> None:
    """ART no modelo completo, um eixo por nivel."""
    dias = sorted(completo["dia"].unique())
    x = np.arange(len(dias))

    for ax, alpha in zip(axes, ALPHAS):
        for termo in ("interacao", "condicao", "genotipo"):
            y = (
                completo.groupby("dia")[coluna_sig(termo, alpha)].mean()
                .reindex(dias).to_numpy() * 100
            )
            ax.plot(x, y, color=COR_TERMO[termo],
                    linewidth=2.6 if termo == "interacao" else 1.7,
                    marker="o", markersize=5, markeredgecolor="white",
                    markeredgewidth=0.7, label=ROTULO_TERMO[termo],
                    zorder=3 if termo == "interacao" else 2)

        ax.axhline(LIMIAR_AGRUPAR * 100, color=COR_TEXTO_FRACO,
                   linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(dias, fontsize=7.5)
        ax.set_ylim(-4, 108)
        ax.set_title(rotulo_alpha(alpha), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.3)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    axes[0].set_ylabel("Bandas significativas (%)", fontsize=9)
    axes[0].text(0.0, 1.11,
                 "A) ART nos 3 genotipos x 2 condicoes: a interacao existe?",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold",
                 ha="left", va="bottom")
    axes[-1].legend(fontsize=7, loc="lower left", framealpha=0.9)


def painel_b(axes: list[plt.Axes], cax: plt.Axes, resumo: pd.DataFrame,
             dias: list[str], pares: pd.DataFrame, rotulos: list[str]) -> None:
    """Interacao par a par, uma matriz por nivel."""
    for ax, alpha in zip(axes, ALPHAS):
        sub = resumo[resumo["alpha"] == alpha]
        grade = grade_por_par(sub, "prop_interacao", dias, pares) * 100

        im = ax.imshow(grade, cmap=CMAP_SEQUENCIAL, vmin=0, vmax=100, aspect="auto")
        anotar_grade(ax, grade, fonte=7.5)

        ax.set_xticks(range(len(dias)))
        ax.set_xticklabels(dias, fontsize=7.5)
        ax.set_yticks(range(len(rotulos)))
        ax.set_yticklabels(rotulos if ax is axes[0] else [], fontsize=8)
        ax.set_title(rotulo_alpha(alpha), fontsize=9, fontweight="bold")
        ax.grid(False)

    axes[0].text(0.0, 1.22,
                 "B) Interacao par a par: qual genotipo a carrega "
                 "(% de bandas com interacao)",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold",
                 ha="left", va="bottom")

    barra = plt.colorbar(im, cax=cax)
    barra.set_label("Bandas com interacao (%)", fontsize=8)
    barra.ax.tick_params(labelsize=7)


def painel_c(axes: list[plt.Axes], cax: plt.Axes, resumo: pd.DataFrame,
             dias: list[str], pares: pd.DataFrame, rotulos: list[str]) -> None:
    """Efeitos simples dentro de cada condicao, no nivel de referencia."""
    sub = resumo[resumo["alpha"] == ALPHA_REFERENCIA]

    for ax, nivel in zip(axes, CONDICOES):
        grade = grade_por_par(sub, f"prop_sig_{nivel}", dias, pares) * 100

        im = ax.imshow(grade, cmap=CMAP_SEQUENCIAL, vmin=0, vmax=100, aspect="auto")
        anotar_grade(ax, grade, fonte=7.5)

        ax.set_xticks(range(len(dias)))
        ax.set_xticklabels(dias, fontsize=7.5)
        ax.set_yticks(range(len(rotulos)))
        ax.set_yticklabels(rotulos if ax is axes[0] else [], fontsize=8)
        ax.set_title(f"somente {nivel}", fontsize=9, fontweight="bold")
        ax.grid(False)

    axes[0].text(0.0, 1.22,
                 "C) Efeitos simples: os genotipos diferem DENTRO de cada "
                 f"condicao?  ({rotulo_alpha(ALPHA_REFERENCIA)})",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold",
                 ha="left", va="bottom")

    barra = plt.colorbar(im, cax=cax)
    barra.set_label("Bandas que separam (%)", fontsize=8)
    barra.ax.tick_params(labelsize=7)


def painel_d(axes: list[plt.Axes], resumo: pd.DataFrame, dias: list[str],
             pares: pd.DataFrame, rotulos: list[str]) -> None:
    """Veredito de agrupamento, uma matriz por nivel."""
    codigo = {v: i for i, v in enumerate(ORDEM_VEREDITO)}
    cmap = LinearSegmentedColormap.from_list(
        "veredito", [COR_VEREDITO[v] for v in ORDEM_VEREDITO], N=len(ORDEM_VEREDITO)
    )

    for ax, alpha in zip(axes, ALPHAS):
        sub = resumo[resumo["alpha"] == alpha]
        grade = np.array([
            sub[(sub["genotipo_a"] == a) & (sub["genotipo_b"] == b)]
            .set_index("dia")["veredito"].reindex(dias).map(codigo).to_numpy()
            for a, b in zip(pares["genotipo_a"], pares["genotipo_b"])
        ], dtype=float)

        ax.imshow(grade, cmap=cmap, vmin=-0.5, vmax=len(ORDEM_VEREDITO) - 0.5,
                  aspect="auto")
        for i in range(grade.shape[0]):
            for j in range(grade.shape[1]):
                rotulo = SIGLA_VEREDITO[ORDEM_VEREDITO[int(grade[i, j])]]
                ax.text(j, i, rotulo, ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white")

        ax.set_xticks(range(len(dias)))
        ax.set_xticklabels(dias, fontsize=7.5)
        ax.set_yticks(range(len(rotulos)))
        ax.set_yticklabels(rotulos if ax is axes[0] else [], fontsize=8)
        ax.set_title(rotulo_alpha(alpha), fontsize=9, fontweight="bold")
        ax.grid(False)

    axes[0].text(0.0, 1.22, "D) Veredito: da para agrupar este par neste dia?",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold",
                 ha="left", va="bottom")

    axes[1].legend(
        handles=[Patch(facecolor=COR_VEREDITO[v],
                       label=f"{SIGLA_VEREDITO[v]} = {v}")
                 for v in ORDEM_VEREDITO],
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
        fontsize=8, frameon=False,
    )


def painel_e(ax: plt.Axes, resumo: pd.DataFrame, dias: list[str],
             pares: pd.DataFrame, rotulos: list[str]) -> None:
    """Quantos dos tres niveis concordam, por par e dia."""
    grade = np.array([
        [
            4 - resumo[(resumo["genotipo_a"] == a) & (resumo["genotipo_b"] == b)
                       & (resumo["dia"] == dia)]["veredito"].nunique()
            for dia in dias
        ]
        for a, b in zip(pares["genotipo_a"], pares["genotipo_b"])
    ], dtype=float)

    ax.imshow(grade, cmap=CMAP_ESTABILIDADE, vmin=0.5, vmax=3.5, aspect="auto")
    for i in range(grade.shape[0]):
        for j in range(grade.shape[1]):
            ax.text(j, i, f"{int(grade[i, j])}/3", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

    ax.set_xticks(range(len(dias)))
    ax.set_xticklabels(dias, fontsize=7.5)
    ax.set_yticks(range(len(rotulos)))
    ax.set_yticklabels(rotulos, fontsize=8)
    ax.set_title("E) Estabilidade: niveis que concordam no veredito\n"
                 "     3/3 = conclusao robusta ao limiar",
                 fontsize=10, fontweight="bold", loc="left")
    ax.grid(False)


def gerar(completo: pd.DataFrame, resumo: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    dias = sorted(resumo["dia"].unique())
    pares = resumo[["genotipo_a", "genotipo_b"]].drop_duplicates()
    rotulos = [f"{a} vs {b}" for a, b in zip(pares["genotipo_a"], pares["genotipo_b"])]

    fig = plt.figure(figsize=(17, 20))
    gs = fig.add_gridspec(
        5, 12, height_ratios=[1.0, 0.62, 0.62, 0.62, 0.62],
        hspace=0.68, wspace=1.5,
        top=0.912, bottom=0.05, left=0.075, right=0.95,
    )

    gs_a = gs[0, :11].subgridspec(1, 3, wspace=0.10)
    axes_a = [fig.add_subplot(gs_a[i]) for i in range(3)]
    painel_a(axes_a, completo)

    gs_b = gs[1, :11].subgridspec(1, 3, wspace=0.10)
    axes_b = [fig.add_subplot(gs_b[i]) for i in range(3)]
    cax_b = fig.add_subplot(gs[1, 11])
    painel_b(axes_b, cax_b, resumo, dias, pares, rotulos)

    gs_d = gs[2, :11].subgridspec(1, 3, wspace=0.10)
    axes_d = [fig.add_subplot(gs_d[i]) for i in range(3)]
    painel_d(axes_d, resumo, dias, pares, rotulos)

    gs_c = gs[3, :11].subgridspec(1, 2, wspace=0.12)
    axes_c = [fig.add_subplot(gs_c[i]) for i in range(2)]
    cax_c = fig.add_subplot(gs[3, 11])
    painel_c(axes_c, cax_c, resumo, dias, pares, rotulos)

    ax_e = fig.add_subplot(gs[4, :6])
    painel_e(ax_e, resumo, dias, pares, rotulos)

    fig.suptitle(
        "Da para agrupar os genotipos? Interacao genotipo x condicao por ART\n"
        "Tres niveis de significancia sobre o q-valor de Benjamini-Hochberg  -  "
        "turno da manha, 192 leituras por dia, 2051 bandas",
        fontsize=14, fontweight="bold", y=0.978,
    )

    fig.text(
        0.075, 0.018,
        "Leitura: apertar o alfa reduz as bandas significativas, e como o "
        "veredito premia a NAO significancia, niveis mais rigorosos empurram "
        "mecanicamente para 'agrupavel'.\nIsso e perda de poder, nao evidencia "
        "de equivalencia. Robusto e o que NAO muda entre os tres niveis -- ver "
        "o painel E.",
        fontsize=8.5, color=COR_TEXTO_FRACO, ha="left", va="bottom",
    )

    fig.savefig(SAIDA, dpi=300)
    print(f"Painel salvo em: {SAIDA}")


def main() -> None:
    print("Gerando painel da interacao genotipo x condicao...\n")
    completo = pd.read_csv(SAIDA_DIR / "interacao_por_banda.csv", sep=";")
    resumo = pd.read_csv(SAIDA_DIR / "interacao_resumo.csv", sep=";")
    print(f"  interacao_por_banda: {len(completo)} linhas")
    print(f"  interacao_resumo: {len(resumo)} linhas "
          f"({len(ALPHAS)} niveis x 3 pares x 7 dias)\n")
    gerar(completo, resumo)
    print("Concluido.")


if __name__ == "__main__":
    main()
