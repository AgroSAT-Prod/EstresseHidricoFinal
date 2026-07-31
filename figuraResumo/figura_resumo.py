#!/usr/bin/env python3
"""Figura-resumo do experimento de estresse hidrico em soja.

Reune numa peca so o que os sete modulos de analise produzem separadamente:
que o espectro separa irrigado de nao irrigado, quando essa separacao aparece,
onde no espectro ela mora, e quais cinco bandas sobrevivem ao funil de selecao.

Paineis:

A) Espectro medio de reflectancia IRRIG vs NIRRIG no dia de maior separacao,
   com as regioes espectrais anotadas e as Top 5 bandas marcadas.
B) PAINEL CENTRAL. Heatmap dia x comprimento de onda do delta de Cliff do
   contraste IRRIG vs NIRRIG, uma faixa por genotipo. Mostra simultaneamente
   onde e quando o estresse aparece, e a escala divergente da a direcao.
C) Trajetoria: % de bandas com IRRIG != NIRRIG ao longo dos sete dias.
D) Matriz 6x6 de separacao entre as celulas genotipo x condicao no dia de pico.
E) Estagios do pre-processamento numa amostra.
F) Reducao de colinearidade: 2051 bandas -> grupos, com o tamanho de cada
   grupo ao longo do espectro.
G) VIP do PLS-DA ao longo do espectro, com a linha VIP = 1, a decisao do
   Boruta codificada e as Top 5 rotuladas.
H) Distribuicao IRRIG vs NIRRIG das cinco bandas selecionadas, no dia de pico.
I) Funil da selecao: 2051 -> representantes -> VIP > 1 -> Boruta -> 5.

Todas as analises usam apenas o turno da manha, o unico presente nos sete dias.

Paleta validada por codigo contra deuteranopia, protanopia e tritanopia --
ver o bloco de cores abaixo para os valores medidos.

Uso:
    python figura_resumo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT.parent / "testeDiferencaSignificativa"))

from shapiro_normalidade import carregar_estagios  # noqa: E402

COMPARACAO = ROOT.parent / "testeDiferencaSignificativa" / "dataset_gerado"
COLINEARIDADE = ROOT.parent / "reducaoColinearidade" / "dataset_gerado"
SELECAO = ROOT.parent / "selecaoVariaveis" / "dataset_gerado"

SAIDA_PNG = ROOT / "figura_resumo.png"
SAIDA_PDF = ROOT / "figura_resumo.pdf"

TURNO = "manha"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]

# Paleta categorica validada em OKLab sob visao normal e as tres dicromacias.
# Os 10 pares passam: pior dE em visao normal 21.6 (piso 15), pior dE sob CVD
# 10.0 em protanopia, no par NIRRIG x BR16 (piso 8). Substitui o par
# verde/vermelho de plot_preprocessamento.py, que protanopia e deuteranopia
# nao distinguem.
COR_CONDICAO = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}

COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"
COR_NEUTRA = "#b8b6b0"

# Divergente para o delta de Cliff: dois polos e cinza neutro exatamente no
# zero. A polaridade repete a da condicao -- laranja para delta positivo, que
# e o lado do NIRRIG.
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "cliff", [COR_CONDICAO["IRRIG"], "#f2f0ed", COR_CONDICAO["NIRRIG"]]
)
# Sequencial de matiz unica para magnitude sem sinal.
CMAP_SEQUENCIAL = LinearSegmentedColormap.from_list(
    "magnitude", ["#f4f6fa", "#7fa8d9", "#1a3a6b"]
)

REGIOES = [
    ("VIS", 400, 700),
    ("red\nedge", 700, 780),
    ("NIR", 780, 1350),
    ("SWIR1", 1350, 1800),
    ("SWIR2", 1800, 2451),
]


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

def dia_de_pico(resumo: pd.DataFrame) -> str:
    """Dia com a maior separacao IRRIG vs NIRRIG entre os pares de estresse."""
    estresse = resumo[resumo["tipo"] == "estresse"]
    return str(estresse.loc[estresse["prop_sig"].idxmax(), "dia"])


# --------------------------------------------------------------------------
# Paineis
# --------------------------------------------------------------------------

def painel_a(
    ax: plt.Axes,
    meta: pd.DataFrame,
    recortado: np.ndarray,
    w: np.ndarray,
    dia: str,
    top5: list[int],
) -> None:
    """Espectro medio de reflectancia por condicao no dia de pico."""
    for condicao in CONDICOES:
        mask = ((meta["dia"] == dia) & (meta["condicao"] == condicao)).to_numpy()
        media = recortado[mask].mean(axis=0)
        desvio = recortado[mask].std(axis=0)
        ax.fill_between(w, media - desvio, media + desvio,
                        color=COR_CONDICAO[condicao], alpha=0.15, linewidth=0)
        ax.plot(w, media, color=COR_CONDICAO[condicao], linewidth=2,
                label=f"{condicao} (n={int(mask.sum())})")

    # Rotulos no rodape e alternados em altura: tres das cinco bandas caem
    # dentro de 36 nm e colidiriam se ficassem na mesma linha.
    for i, banda in enumerate(sorted(top5)):
        ax.axvline(banda, color=COR_TEXTO_FRACO, linestyle=":", linewidth=1,
                   alpha=0.7, zorder=0)
        ax.annotate(f"{banda}", xy=(banda, 0.10 + 0.07 * (i % 2)),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=COR_TEXTO,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))

    anotar_regioes(ax)
    ax.set_xlim(w.min(), w.max())
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("Reflectancia", fontsize=9)
    ax.set_title(
        f"A) Espectro medio por condicao em {dia}, o dia de maior separacao  "
        "(faixa: +- 1 desvio padrao; tracejado: Top 5 bandas)",
        fontsize=10, fontweight="bold", loc="left",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def anotar_regioes(ax: plt.Axes, com_rotulo: bool = True) -> None:
    """Sombreia e nomeia as regioes espectrais."""
    for i, (nome, ini, fim) in enumerate(REGIOES):
        if i % 2:
            ax.axvspan(ini, fim, color=COR_NEUTRA, alpha=0.12, zorder=0)
        if com_rotulo:
            ax.annotate(nome, xy=((ini + fim) / 2, 0.02), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5,
                        color=COR_TEXTO_FRACO, fontweight="bold")


def painel_b(
    axes: list[plt.Axes],
    estresse: pd.DataFrame,
    cax: plt.Axes,
    w: np.ndarray,
) -> None:
    """Heatmap dia x comprimento de onda do delta de Cliff, um por genotipo."""
    dias = sorted(estresse["dia"].unique())

    for ax, genotipo in zip(axes, GENOTIPOS):
        sub = estresse[estresse["genotipo"] == genotipo]
        grade = (
            sub.pivot(index="dia", columns="banda_nm", values="delta_cliff")
            .reindex(dias)
        )

        im = ax.imshow(
            grade.to_numpy(), aspect="auto", cmap=CMAP_DIVERGENTE,
            norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
            extent=[w.min(), w.max(), len(dias) - 0.5, -0.5],
            interpolation="nearest",
        )
        ax.set_yticks(range(len(dias)))
        ax.set_yticklabels(dias, fontsize=7.5)
        ax.set_ylabel(genotipo, fontsize=9, fontweight="bold",
                      color=COR_GENOTIPO[genotipo])
        ax.tick_params(labelsize=7.5)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
        for _, ini, fim in REGIOES:
            ax.axvline(ini, color="white", linewidth=0.6, alpha=0.5)

    axes[-1].set_xlabel("Comprimento de onda (nm)", fontsize=9)
    axes[0].set_title(
        "B) Delta de Cliff do contraste IRRIG vs NIRRIG: onde e quando o "
        "estresse aparece no espectro\n"
        "     Escala SNV, que mede forma e nao magnitude -- o sinal NAO e "
        "comparavel ao do painel A (em 900 nm o NIRRIG tem reflectancia maior, "
        "mas valor SNV menor)",
        fontsize=10, fontweight="bold", loc="left",
    )

    barra = plt.colorbar(im, cax=cax)
    barra.set_label("delta de Cliff  (+ : NIRRIG acima)", fontsize=8)
    barra.ax.tick_params(labelsize=7)


def painel_c(ax: plt.Axes, estresse: pd.DataFrame) -> None:
    """Trajetoria do efeito do estresse ao longo dos dias, por genotipo."""
    dias = sorted(estresse["dia"].unique())
    x = np.arange(len(dias))

    for genotipo in GENOTIPOS:
        sub = estresse[estresse["genotipo"] == genotipo]
        prop = (
            sub.groupby("dia")["significativa"].mean().reindex(dias).to_numpy() * 100
        )
        ax.plot(x, prop, color=COR_GENOTIPO[genotipo], linewidth=2,
                marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=0.8, label=genotipo)
        # Rotulo direto: identidade nao depende so da cor.
        ax.annotate(genotipo, xy=(x[-1], prop[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    fontweight="bold", color=COR_GENOTIPO[genotipo])

    ax.set_xticks(x)
    ax.set_xticklabels(dias, fontsize=8)
    ax.set_xlim(-0.3, len(dias) - 0.3)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("Dia de coleta", fontsize=9)
    ax.set_ylabel("Bandas com IRRIG != NIRRIG (%)", fontsize=9)
    ax.set_title("C) Trajetoria do efeito do estresse", fontsize=10,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right", title="Genotipo", title_fontsize=8)
    ax.grid(True, alpha=0.3)


def painel_d(ax: plt.Axes, resumo: pd.DataFrame, cax: plt.Axes, dia: str) -> None:
    """Matriz 6x6 de separacao entre as celulas genotipo x condicao."""
    celulas = [f"{g}|{c}" for g in GENOTIPOS for c in CONDICOES]
    grade = np.full((len(celulas), len(celulas)), np.nan)

    sub = resumo[resumo["dia"] == dia]
    indice = {nome: i for i, nome in enumerate(celulas)}
    for _, row in sub.iterrows():
        i, j = indice[row["celula_a"]], indice[row["celula_b"]]
        grade[i, j] = grade[j, i] = row["prop_sig"] * 100

    im = ax.imshow(grade, cmap=CMAP_SEQUENCIAL, vmin=0, vmax=100)

    for i in range(len(celulas)):
        for j in range(len(celulas)):
            if np.isnan(grade[i, j]):
                continue
            claro = grade[i, j] > 55
            ax.text(j, i, f"{grade[i, j]:.0f}", ha="center", va="center",
                    fontsize=7.5, color="white" if claro else COR_TEXTO)

    rotulos = [c.replace("|", "\n") for c in celulas]
    ax.set_xticks(range(len(celulas)))
    ax.set_yticks(range(len(celulas)))
    ax.set_xticklabels(rotulos, fontsize=7)
    ax.set_yticklabels(rotulos, fontsize=7)
    for i, nome in enumerate(celulas):
        cor = COR_GENOTIPO[nome.split("|")[0]]
        ax.get_xticklabels()[i].set_color(cor)
        ax.get_yticklabels()[i].set_color(cor)

    ax.set_title(f"D) Separacao entre as celulas em {dia}", fontsize=10,
                 fontweight="bold", loc="left")
    ax.grid(False)
    barra = plt.colorbar(im, cax=cax)
    barra.set_label("Bandas separadas (%)", fontsize=8)
    barra.ax.tick_params(labelsize=7)


def painel_e(ax: plt.Axes, estagios: dict[str, np.ndarray], w: np.ndarray) -> None:
    """Estagios do pre-processamento numa unica amostra."""
    # Eixos independentes empilhados seriam dois eixos y no mesmo grafico; em
    # vez disso cada estagio e deslocado verticalmente e o eixo y vira
    # qualitativo, com a escala de cada um anotada no proprio rotulo.
    rotulos = [
        ("recortado", "Recorte +\njump correction"),
        ("suavizado", "+ Savitzky-\nGolay"),
        ("normalizado", "+ SNV"),
    ]
    deslocamento = 0.0
    posicoes, nomes = [], []
    for i, (chave, nome) in enumerate(rotulos):
        y = estagios[chave][0]
        y = (y - y.mean()) / (y.std() if y.std() else 1.0)
        ax.plot(w, y + deslocamento, linewidth=1.1,
                color=[COR_CONDICAO["IRRIG"], "#7fa8d9", COR_TEXTO_FRACO][i])
        posicoes.append(deslocamento)
        nomes.append(nome)
        deslocamento -= 7.0

    ax.set_yticks(posicoes)
    ax.set_yticklabels(nomes, fontsize=8)
    ax.set_xlim(w.min(), w.max())
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_title("E) Estagios do pre-processamento (uma amostra, cada estagio\n"
                 "padronizado para caber na mesma escala)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, axis="x")


def painel_f(ax: plt.Axes, grupos: pd.DataFrame, n_bandas: int) -> None:
    """Colinearidade interna dos grupos ao longo do espectro.

    O tamanho dos grupos nao vale um grafico: praticamente todos batem no teto
    de 10 bandas, porque a janela de 10 nm e que limita, nao o criterio de
    correlacao -- bandas vizinhas de 1 nm quase sempre passam de |r| = 0.80.
    O que varia, e informa, e a correlacao *dentro* de cada grupo: onde ela
    cai, o espectro tem estrutura fina que a janela nao consegue colapsar.
    """
    centro = (grupos["inicio_nm"] + grupos["fim_nm"]) / 2

    ax.plot(centro, grupos["r_abs_medio_grupo"], color=COR_CONDICAO["IRRIG"],
            linewidth=1.6, label="|r| medio no grupo")
    ax.fill_between(centro, grupos["r_abs_min_grupo"],
                    grupos["r_abs_medio_grupo"],
                    color=COR_CONDICAO["IRRIG"], alpha=0.2, linewidth=0,
                    label="ate o |r| minimo do grupo")
    ax.axhline(0.80, color=COR_CONDICAO["NIRRIG"], linestyle="--",
               linewidth=1.2, label="Limiar |r| = 0.80")

    anotar_regioes(ax, com_rotulo=False)
    reducao = 1 - len(grupos) / n_bandas
    ax.set_xlim(centro.min(), centro.max())
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("|Spearman| dentro do grupo", fontsize=9)
    ax.set_title(
        f"F) Reducao de colinearidade: {n_bandas} bandas -> {len(grupos)} "
        f"grupos de ate 10 nm ({reducao:.0%})",
        fontsize=10, fontweight="bold", loc="left",
    )
    ax.legend(fontsize=7.5, loc="lower left", ncol=3)
    ax.grid(True, alpha=0.3)


def painel_g(ax: plt.Axes, selecao: pd.DataFrame, top5: list[int]) -> None:
    """VIP ao longo do espectro, com decisao do Boruta e as Top 5."""
    confirmada = selecao["confirmada_boruta"].to_numpy()

    ax.scatter(selecao.loc[~confirmada, "banda_nm"], selecao.loc[~confirmada, "vip"],
               s=18, color=COR_NEUTRA, edgecolor="white", linewidth=0.4,
               label="Boruta: rejeitada ou indefinida", zorder=2)
    ax.scatter(selecao.loc[confirmada, "banda_nm"], selecao.loc[confirmada, "vip"],
               s=26, color=COR_CONDICAO["IRRIG"], edgecolor="white",
               linewidth=0.4, label="Boruta: confirmada", zorder=3)

    ax.axhline(1.0, color=COR_CONDICAO["NIRRIG"], linestyle="--", linewidth=1.4,
               label="VIP = 1 (corte)", zorder=1)

    escolhidas = selecao[selecao["banda_nm"].isin(top5)]
    ax.scatter(escolhidas["banda_nm"], escolhidas["vip"], s=130, marker="*",
               color=COR_CONDICAO["NIRRIG"], edgecolor="white", linewidth=0.8,
               label="Top 5", zorder=4)
    for _, row in escolhidas.iterrows():
        ax.annotate(f"{int(row['banda_nm'])} nm",
                    xy=(row["banda_nm"], row["vip"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8,
                    fontweight="bold", color=COR_TEXTO)

    anotar_regioes(ax, com_rotulo=False)
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("VIP (PLS-DA)", fontsize=9)
    ax.set_title("G) Selecao de variaveis: VIP do PLS-DA e decisao do Boruta "
                 "nas bandas representativas",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)


def painel_h(
    axes: list[plt.Axes],
    meta: pd.DataFrame,
    normalizado: np.ndarray,
    w: np.ndarray,
    dia: str,
    top5: list[int],
) -> None:
    """Distribuicao IRRIG vs NIRRIG das cinco bandas escolhidas."""
    for ax, banda in zip(axes, top5):
        j = int(np.flatnonzero(w == banda)[0])
        dados, cores = [], []
        for condicao in CONDICOES:
            mask = ((meta["dia"] == dia) & (meta["condicao"] == condicao)).to_numpy()
            dados.append(normalizado[mask, j])
            cores.append(COR_CONDICAO[condicao])

        partes = ax.violinplot(dados, positions=[0, 1], widths=0.7,
                               showmeans=False, showextrema=False)
        for corpo, cor in zip(partes["bodies"], cores):
            corpo.set_facecolor(cor)
            corpo.set_alpha(0.55)
            corpo.set_edgecolor("white")
            corpo.set_linewidth(1.0)

        caixas = ax.boxplot(dados, positions=[0, 1], widths=0.16,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="white", linewidth=1.4))
        for caixa, cor in zip(caixas["boxes"], cores):
            caixa.set_facecolor(cor)
            caixa.set_edgecolor("none")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(CONDICOES, fontsize=7.5)
        ax.set_title(f"{banda} nm", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Reflectancia (SNV)", fontsize=9)
    # O titulo do painel vai acima dos titulos individuais das cinco bandas,
    # que ja ocupam o slot de title de cada eixo.
    axes[0].text(
        0.0, 1.22, f"H) Distribuicao das Top 5 bandas em {dia}",
        transform=axes[0].transAxes, fontsize=10, fontweight="bold",
        ha="left", va="bottom",
    )


def painel_i(ax: plt.Axes, etapas: list[tuple[str, int]]) -> None:
    """Funil da selecao de variaveis."""
    nomes = [nome for nome, _ in etapas]
    valores = [valor for _, valor in etapas]
    y = np.arange(len(etapas))

    cores = CMAP_SEQUENCIAL(np.linspace(0.25, 0.95, len(etapas)))
    ax.barh(y, valores, color=cores, height=0.62, linewidth=0)

    for i, valor in enumerate(valores):
        ax.annotate(f"{valor}", xy=(valor, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    fontweight="bold", color=COR_TEXTO)

    ax.set_yticks(y)
    ax.set_yticklabels(nomes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, valores[0] * 2.2)
    ax.set_xlabel("Bandas (escala log)", fontsize=9)
    ax.set_title("I) Funil da selecao", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, axis="x")


# --------------------------------------------------------------------------

def gerar(
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
    estresse: pd.DataFrame,
    resumo: pd.DataFrame,
    grupos: pd.DataFrame,
    selecao: pd.DataFrame,
    top5: list[int],
    dia: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(18, 24))
    # top/bottom explicitos: sem eles o gridspec deixa uma faixa vazia entre o
    # suptitle e o painel A, que o bbox_inches="tight" preserva.
    gs = fig.add_gridspec(
        6, 12,
        height_ratios=[1.0, 1.45, 1.0, 1.0, 1.0, 0.95],
        hspace=0.62, wspace=1.6,
        top=0.955, bottom=0.045, left=0.065, right=0.945,
    )

    ax_a = fig.add_subplot(gs[0, :])
    painel_a(ax_a, meta, estagios["recortado"], w, dia, top5)

    gs_b = gs[1, :11].subgridspec(3, 1, hspace=0.12)
    axes_b = [fig.add_subplot(gs_b[i]) for i in range(3)]
    cax_b = fig.add_subplot(gs[1, 11])
    painel_b(axes_b, estresse, cax_b, w)

    ax_c = fig.add_subplot(gs[2, :6])
    painel_c(ax_c, estresse)

    ax_d = fig.add_subplot(gs[2, 7:11])
    cax_d = fig.add_subplot(gs[2, 11])
    painel_d(ax_d, resumo, cax_d, dia)

    ax_e = fig.add_subplot(gs[3, :6])
    painel_e(ax_e, estagios, w)

    ax_f = fig.add_subplot(gs[3, 6:])
    painel_f(ax_f, grupos, len(w))

    ax_g = fig.add_subplot(gs[4, :])
    painel_g(ax_g, selecao, top5)

    gs_h = gs[5, :7].subgridspec(1, 5, wspace=0.35)
    axes_h = [fig.add_subplot(gs_h[i]) for i in range(5)]
    painel_h(axes_h, meta, estagios["normalizado"], w, dia, top5)

    ax_i = fig.add_subplot(gs[5, 8:])
    etapas = [
        ("Espectro completo", len(w)),
        ("Representantes\n(colinearidade)", len(grupos)),
        ("VIP > 1", int(selecao["vip_acima_de_1"].sum())),
        ("Confirmadas\npelo Boruta", int(
            (selecao["vip_acima_de_1"] & selecao["confirmada_boruta"]).sum())),
        ("Top 5", len(top5)),
    ]
    painel_i(ax_i, etapas)

    # Legenda unica da condicao, ancorada no rodape: a mesma codificacao de cor
    # vale para os paineis A, E e H.
    fig.legend(
        handles=[Patch(facecolor=COR_CONDICAO[c], label=c) for c in CONDICOES]
        + [Line2D([0], [0], color=COR_GENOTIPO[g], linewidth=2.5, label=g)
           for g in GENOTIPOS],
        loc="lower center", ncol=5, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        "Estresse hidrico em soja: deteccao espectral por genotipo e dia\n"
        f"1348 leituras hiperespectrais (400-2450 nm), turno da manha, "
        f"3 genotipos x 2 condicoes x 4 blocos x 7 dias",
        fontsize=16, fontweight="bold", y=0.988,
    )

    for saida in (SAIDA_PNG, SAIDA_PDF):
        fig.savefig(saida, dpi=300)
        print(f"Figura salva em: {saida}")


def main() -> None:
    print("Carregando espectro e derivando os estagios...")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")

    estresse = pd.read_csv(COMPARACAO / "comparacao_estresse.csv", sep=";")
    resumo = pd.read_csv(COMPARACAO / "comparacao_resumo.csv", sep=";")
    grupos = pd.read_csv(COLINEARIDADE / "bandas_representativas.csv", sep=";")
    selecao = pd.read_csv(SELECAO / "selecao_variaveis.csv", sep=";")
    top5 = pd.read_csv(SELECAO / "top5_bandas.csv", sep=";")

    bandas_top5 = [int(b) for b in top5.sort_values("posicao")["banda_nm"]]
    dia = dia_de_pico(resumo)
    print(f"  dia de maior separacao IRRIG vs NIRRIG: {dia}")
    print(f"  Top 5 bandas: {bandas_top5}")

    print("Gerando a figura...")
    gerar(meta, estagios, w, estresse, resumo, grupos, selecao, bandas_top5, dia)
    print("Concluido.")


if __name__ == "__main__":
    main()
