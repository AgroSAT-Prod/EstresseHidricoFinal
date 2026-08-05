#!/usr/bin/env python3
"""Como as cinco bandas otimas foram escolhidas: os criterios, lado a lado.

Companheira de `plot_bandas_otimas.py`, que mostra o que as bandas fazem nos
dados. Esta mostra por que sao elas, e nao outras -- os dois criterios de
importancia (VIP e Boruta) ao longo do espectro, a validacao que fixou o
tamanho do modelo PLS, e o funil que vai de 2051 bandas a 5.

Paineis:

A) VIP do PLS-DA ao longo do espectro, nas 206 bandas representativas. A linha
   VIP = 1 e o corte usual: como a media dos VIP ao quadrado vale 1 por
   construcao, acima dela a banda contribui acima da media. A cor codifica a
   decisao do Boruta, entao os dois criterios se leem de uma vez.
B) Importancia media do Boruta ao longo do espectro, na mesma escala espectral
   de A. A altura e a importancia da Random Forest; a cor, a decisao.
C) Os dois criterios um contra o outro. As candidatas sao os pontos azuis
   acima da linha VIP = 1 -- o Boruta nao tem limiar num eixo, a decisao dele
   esta na cor. Os dois criterios concordam so em parte (Spearman +0.56), e e
   por isso que o desempate usa a media das posicoes nos dois rankings, e nao
   um so: as bandas de VIP mais alto nao sao as de maior importancia na
   Random Forest.
D) Validacao cruzada por bloco do PLS-DA. As 8 leituras de uma planta sao
   quase duplicatas, entao as dobras respeitam os 4 blocos de campo: dividir
   ao acaso deixaria a mesma planta nos dois lados e inflaria a acuracia.
E) Funil da selecao, em escala log.
F) Significancia das Top 5 nos dois testes que alimentam o criterio 1: efeito
   de condicao no teste por dia, e efeito de tempo na analise temporal.

Ressalva do funil
-----------------
Todas as 206 representantes passam no criterio de significancia -- ele nao
elimina ninguem neste dataset. Nao e um erro do filtro: e a pseudorreplicacao
dos outros modulos aparecendo aqui. Com 32 leituras por celula e apenas 4
blocos independentes, quase toda banda da q < 0.05. Quem realmente estreita o
funil sao VIP e Boruta, e o painel E mostra isso pela ausencia de degrau.

Uso:
    python plot_bandas_otimas_criterios.py
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

from plot_bandas_otimas import REGIOES, regiao_de  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"

# A selecao e a reducao de colinearidade rodam por genotipo; a figura mostra o
# funil de um deles. CD202 e o de resposta mais forte ao estresse.
GENOTIPO = "CD202"
SAIDA = ROOT / "bandas_otimas_criterios.png"

REPRESENTANTES = (
    ROOT.parent / "reducaoColinearidade" / "dataset_gerado" / GENOTIPO
    / "bandas_representativas.csv"
)

N_BANDAS_ESPECTRO = 2051

VIP_MIN = 1.0
ALPHA = 0.05

COR_DECISAO = {
    "confirmada": "#2a78d6",
    "tentativa": "#b8b6b0",
    "rejeitada": "#eb6834",
}
ORDEM_DECISAO = ["confirmada", "tentativa", "rejeitada"]

COR_TOP5 = "#1a1a19"
COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"
COR_NEUTRA = "#b8b6b0"
COR_LIMIAR = "#c2410c"

COR_CRITERIO = {"condicao": "#2a78d6", "tempo": "#7b4bb5"}


def sombrear_regioes(ax: plt.Axes, rotular: bool) -> None:
    """Faixas alternadas das regioes espectrais, para orientar a leitura."""
    for i, (nome, inicio, fim) in enumerate(REGIOES):
        if i % 2 == 0:
            ax.axvspan(inicio, fim, color=COR_NEUTRA, alpha=0.12, zorder=0)
        if rotular:
            ax.text((inicio + fim) / 2, 1.02, nome,
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=8, fontweight="bold",
                    color=COR_TEXTO_FRACO)


def rotular_top5(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    rotulos: list[str],
    deslocamento: tuple[int, int] = (0, 12),
) -> None:
    """Marca e rotula as Top 5 sobre a serie do painel.

    O rotulo vem de fora porque o eixo x muda de significado entre os paineis:
    nos espectrais ele e o comprimento de onda, no painel C e a importancia do
    Boruta.
    """
    ax.scatter(x, y, s=62, marker="D", facecolor="none", edgecolor=COR_TOP5,
               linewidth=1.4, zorder=5)
    for xi, yi, rotulo in zip(x, y, rotulos):
        ax.annotate(rotulo, xy=(xi, yi), xytext=deslocamento,
                    textcoords="offset points", ha="center", fontsize=8,
                    fontweight="bold", color=COR_TEXTO, zorder=6)


def rotulos_top5(ordem: pd.DataFrame, com_banda: bool = True) -> list[str]:
    """`#posicao  banda nm`, ou so a posicao quando o eixo ja diz a banda."""
    return [
        f"#{int(p)}  {int(b)}" if com_banda else f"#{int(p)}"
        for p, b in zip(ordem["posicao"], ordem["banda_nm"])
    ]


def painel_a(ax: plt.Axes, selecao: pd.DataFrame, top5: pd.DataFrame) -> None:
    """VIP ao longo do espectro, com a decisao do Boruta na cor."""
    sombrear_regioes(ax, rotular=True)

    ax.vlines(selecao["banda_nm"], VIP_MIN, selecao["vip"],
              color=COR_NEUTRA, linewidth=0.6, alpha=0.5, zorder=1)
    for decisao in ORDEM_DECISAO:
        sub = selecao[selecao["decisao_boruta"] == decisao]
        ax.scatter(sub["banda_nm"], sub["vip"], s=30,
                   color=COR_DECISAO[decisao], edgecolor="white",
                   linewidth=0.5, zorder=3, label=decisao)

    ax.axhline(VIP_MIN, color=COR_LIMIAR, linestyle="--", linewidth=1.4,
               zorder=2)
    ax.text(2440, VIP_MIN, " VIP = 1", va="bottom", ha="right", fontsize=8.5,
            fontweight="bold", color=COR_LIMIAR)

    escolhidas = selecao[selecao["banda_nm"].isin(top5["banda_nm"])]
    ordem = escolhidas.merge(top5[["banda_nm", "posicao"]], on="banda_nm")
    rotular_top5(ax, ordem["banda_nm"].to_numpy(), ordem["vip"].to_numpy(),
                 rotulos_top5(ordem))

    n_acima = int(selecao["vip_acima_de_1"].sum())
    ax.set_xlim(385, 2465)
    ax.set_ylim(0, max(selecao["vip"].max(), 2.2) * 1.18)
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("VIP", fontsize=9)
    ax.set_title(
        f"A) VIP do PLS-DA nas {len(selecao)} bandas representativas  -  "
        f"{n_acima} com VIP > 1",
        fontsize=11, fontweight="bold", loc="left", pad=22,
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def painel_b(ax: plt.Axes, selecao: pd.DataFrame, top5: pd.DataFrame) -> None:
    """Importancia media do Boruta ao longo do espectro."""
    sombrear_regioes(ax, rotular=False)

    for decisao in ORDEM_DECISAO:
        sub = selecao[selecao["decisao_boruta"] == decisao]
        ax.bar(sub["banda_nm"], sub["importancia_media"], width=7,
               color=COR_DECISAO[decisao], linewidth=0, zorder=2)

    escolhidas = selecao[selecao["banda_nm"].isin(top5["banda_nm"])]
    ordem = escolhidas.merge(top5[["banda_nm", "posicao"]], on="banda_nm")
    rotular_top5(ax, ordem["banda_nm"].to_numpy(),
                 ordem["importancia_media"].to_numpy(),
                 rotulos_top5(ordem))

    n_confirmadas = int((selecao["decisao_boruta"] == "confirmada").sum())
    ax.set_xlim(385, 2465)
    ax.set_ylim(0, selecao["importancia_media"].max() * 1.22)
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("Importancia media (Random Forest)", fontsize=9)
    ax.set_title(
        f"B) Importancia do Boruta ao longo do espectro  -  "
        f"{n_confirmadas} bandas confirmadas contra as sombras",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def painel_c(ax: plt.Axes, selecao: pd.DataFrame, top5: pd.DataFrame) -> None:
    """VIP contra importancia do Boruta: onde os dois criterios concordam."""
    for decisao in ORDEM_DECISAO:
        sub = selecao[selecao["decisao_boruta"] == decisao]
        ax.scatter(sub["importancia_media"], sub["vip"], s=26,
                   color=COR_DECISAO[decisao], edgecolor="white",
                   linewidth=0.4, alpha=0.9, zorder=2)

    ax.axhline(VIP_MIN, color=COR_LIMIAR, linestyle="--", linewidth=1.2,
               zorder=1)

    ordem = selecao.merge(top5[["banda_nm", "posicao"]], on="banda_nm")
    rotular_top5(ax, ordem["importancia_media"].to_numpy(),
                 ordem["vip"].to_numpy(), rotulos_top5(ordem),
                 deslocamento=(0, 11))

    # Spearman, e nao Pearson: o que interessa e se os dois rankings concordam,
    # e as duas escalas nao sao comparaveis nem lineares entre si.
    rho = selecao[["vip", "importancia_media"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.96, f"Spearman entre os dois criterios: {rho:+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=COR_TEXTO_FRACO)

    ax.set_xscale("log")
    ax.set_xlabel("Importancia media do Boruta (escala log)", fontsize=9)
    ax.set_ylabel("VIP", fontsize=9)
    ax.set_title("C) Os dois criterios de importancia, um contra o outro",
                 fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def painel_d(ax: plt.Axes, validacao: pd.DataFrame) -> None:
    """Acuracia da validacao cruzada por bloco, por numero de componentes."""
    x = validacao["n_componentes"].to_numpy()
    y = validacao["acuracia_cv"].to_numpy()
    desvio = validacao["desvio_cv"].to_numpy()

    ax.fill_between(x, y - desvio, y + desvio, color=COR_CRITERIO["condicao"],
                    alpha=0.18, linewidth=0)
    ax.plot(x, y, color=COR_CRITERIO["condicao"], linewidth=1.8, marker="o",
            markersize=4)

    melhor = int(x[int(np.argmax(y))])
    ax.axvline(melhor, color=COR_LIMIAR, linestyle="--", linewidth=1.3)
    ax.annotate(
        f"{melhor} componentes\nacuracia {y.max():.3f}",
        xy=(melhor, y.max()), xytext=(-12, -34), textcoords="offset points",
        ha="right", fontsize=8.5, fontweight="bold", color=COR_TEXTO,
    )

    ax.set_xticks(x[::2])
    ax.set_xlabel("Componentes do PLS", fontsize=9)
    ax.set_ylabel("Acuracia (CV por bloco)", fontsize=9)
    ax.set_title("D) Quantos componentes o PLS-DA usa",
                 fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def painel_e(ax: plt.Axes, etapas: list[tuple[str, int]]) -> None:
    """Funil da selecao, do espectro inteiro as cinco bandas."""
    nomes = [nome for nome, _ in etapas]
    valores = [valor for _, valor in etapas]
    y = np.arange(len(etapas))

    cores = plt.get_cmap("Blues")(np.linspace(0.85, 0.35, len(etapas)))
    ax.barh(y, valores, color=cores, height=0.62, linewidth=0)

    anterior = None
    for i, valor in enumerate(valores):
        texto = f"{valor}"
        if anterior is not None:
            texto += (f"   ({valor / anterior:.0%} do passo anterior)"
                      if valor != anterior else "   (nao elimina nenhuma)")
        ax.annotate(texto, xy=(valor, i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    fontweight="bold", color=COR_TEXTO)
        anterior = valor

    ax.set_yticks(y)
    ax.set_yticklabels(nomes, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, valores[0] * 9)
    ax.set_xlabel("Bandas (escala log)", fontsize=9)
    ax.set_title("E) Funil da selecao", fontsize=11, fontweight="bold",
                 loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, axis="x")


def painel_f(ax: plt.Axes, top5: pd.DataFrame) -> None:
    """Significancia das Top 5 nos dois testes que alimentam o criterio 1."""
    ordem = top5.sort_values("posicao")
    x = np.arange(len(ordem))
    largura = 0.38

    for i, (coluna, rotulo) in enumerate(
        [("q_condicao", "condicao"), ("q_tempo", "tempo")]
    ):
        with np.errstate(divide="ignore"):
            altura = -np.log10(ordem[coluna].to_numpy())
        ax.bar(x + (i - 0.5) * largura, altura, width=largura,
               color=COR_CRITERIO[rotulo], alpha=0.85, linewidth=0,
               label=f"efeito de {rotulo}")

    ax.axhline(-np.log10(ALPHA), color=COR_LIMIAR, linestyle="--",
               linewidth=1.3)
    ax.text(-0.42, -np.log10(ALPHA), f"q = {ALPHA:g} ",
            va="bottom", ha="left", fontsize=8.5, fontweight="bold",
            color=COR_LIMIAR)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"#{int(p)}\n{int(b)} nm\n{regiao_de(int(b))}"
         for p, b in zip(ordem["posicao"], ordem["banda_nm"])],
        fontsize=8.5,
    )
    ax.set_ylabel("-log10(q)", fontsize=9)
    ax.set_title("F) Significancia das Top 5 nos dois testes de origem",
                 fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)


def gerar(
    selecao: pd.DataFrame,
    top5: pd.DataFrame,
    validacao: pd.DataFrame,
    n_representantes: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(
        4, 2, hspace=0.42, wspace=0.20,
        top=0.915, bottom=0.055, left=0.075, right=0.975,
        height_ratios=[1.15, 0.95, 1.0, 1.0],
    )

    painel_a(fig.add_subplot(gs[0, :]), selecao, top5)
    painel_b(fig.add_subplot(gs[1, :]), selecao, top5)
    painel_c(fig.add_subplot(gs[2, 0]), selecao, top5)
    painel_d(fig.add_subplot(gs[2, 1]), validacao)

    etapas = [
        ("Espectro completo", N_BANDAS_ESPECTRO),
        ("Representantes\n(colinearidade)", n_representantes),
        ("Significativas\n(condicao ou tempo)", int(selecao["significativa"].sum())),
        ("VIP > 1", int((selecao["significativa"]
                         & selecao["vip_acima_de_1"]).sum())),
        ("Confirmadas\npelo Boruta", int((selecao["significativa"]
                                          & selecao["vip_acima_de_1"]
                                          & selecao["confirmada_boruta"]).sum())),
        ("Top 5", len(top5)),
    ]
    painel_e(fig.add_subplot(gs[3, 0]), etapas)
    painel_f(fig.add_subplot(gs[3, 1]), top5)

    fig.suptitle(
        f"Como as cinco bandas otimas de {GENOTIPO} foram escolhidas: VIP, "
        "Boruta, significancia e o funil\n"
        "Alvo: IRRIG vs NIRRIG  -  espectro normalizado (SNV), turno da manha",
        fontsize=15, fontweight="bold", y=0.972,
    )

    fig.legend(
        handles=[Patch(facecolor=COR_DECISAO[d], label=f"Boruta: {d}")
                 for d in ORDEM_DECISAO]
        + [Line2D([0], [0], marker="D", markerfacecolor="none",
                  markeredgecolor=COR_TOP5, markersize=8, linewidth=0,
                  label="Top 5"),
           Line2D([0], [0], color=COR_LIMIAR, linestyle="--", linewidth=1.4,
                  label="Limiar do criterio")],
        loc="lower center", ncol=5, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.016),
    )

    fig.text(
        0.075, 0.004,
        "O passo de significancia nao elimina nenhuma banda: com 32 leituras "
        "por celula e 4 blocos independentes, quase toda banda da q < 0.05. "
        "Quem estreita o funil sao VIP e Boruta.",
        fontsize=9, color=COR_TEXTO_FRACO, ha="left", va="bottom",
    )

    fig.savefig(SAIDA, dpi=180)
    print(f"Figura salva em: {SAIDA}")


def main() -> None:
    print("Gerando os criterios de selecao das bandas otimas...\n")

    selecao = pd.read_csv(SAIDA_DIR / GENOTIPO / "selecao_variaveis.csv", sep=";")
    top5 = pd.read_csv(SAIDA_DIR / GENOTIPO / "top5_bandas.csv", sep=";")
    validacao = pd.read_csv(SAIDA_DIR / GENOTIPO / "pls_da_validacao.csv", sep=";")
    representantes = pd.read_csv(REPRESENTANTES, sep=";")

    print(f"  {len(selecao)} bandas representativas avaliadas")
    print(f"  VIP > 1: {int(selecao['vip_acima_de_1'].sum())}   "
          f"confirmadas pelo Boruta: "
          f"{int(selecao['confirmada_boruta'].sum())}")
    print(f"  Top 5: {', '.join(str(int(b)) for b in top5['banda_nm'])}\n")

    gerar(selecao, top5, validacao, len(representantes))
    print("Concluido.")


if __name__ == "__main__":
    main()
