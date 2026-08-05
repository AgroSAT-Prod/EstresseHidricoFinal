#!/usr/bin/env python3
"""O que a reducao de colinearidade faz com o espectro, genotipo a genotipo.

`reducao_colinearidade.py` diz quantos grupos sobraram; esta figura mostra o
que se perde e o que se mantem ao trocar 2051 bandas por 206 representantes.
Um bloco por genotipo -- as tres analises sao independentes, com janelas
iguais mas representantes diferentes.

Cada bloco tem duas linhas:

1. VISAO GERAL. O espectro medio do genotipo em 400-2450 nm duas vezes: a
   curva "antes", sobre as 2051 bandas, e a curva "depois", so pelos 206
   representantes. O titulo traz o erro de reconstrucao -- o desvio entre
   interpolar os representantes de volta para a grade cheia e a curva
   original. E o numero que sustenta a leitura visual de que a forma
   sobrevive a 10% das bandas.
2. RECORTES. Cinco ampliacoes de 60 nm, uma por banda da Top 5 daquele
   genotipo. So nessa escala as janelas de 10 nm aparecem: no eixo inteiro
   elas ficam menores que a espessura da linha. Dentro de cada recorte estao
   as faixas das janelas, as 10 bandas de cada uma e o ponto principal --
   o representante que a janela elegeu.

Por que a Top 5 centra os recortes
----------------------------------
A Top 5 vem de `selecaoVariaveis`, que roda *depois* desta etapa e so pode
escolher entre representantes. Os recortes mostram, entao, o elo entre as duas:
qual banda cada janela decisiva elegeu, e quais nove ela descartou. Sem o
arquivo da selecao, os recortes caem para os cinco representantes de menor q.

Escala vertical
---------------
As curvas estao em unidades de SNV, o estagio em que a reducao rodou -- nao em
reflectancia. E a escala sobre a qual a correlacao de Spearman foi calculada,
entao e a que corresponde ao agrupamento desenhado.

Uso:
    python plot_reducao_colinearidade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402

DADOS = ROOT / "dataset_gerado"
SELECAO = ROOT.parent / "selecaoVariaveis" / "dataset_gerado"
SAIDA = ROOT / "reducao_colinearidade.png"

ESTAGIO = "normalizado"
TURNO = "manha"

GENOTIPOS = ["BR16", "CD202", "EMB48"]

# Meia largura dos recortes: 60 nm mostram seis janelas de 10 nm, o bastante
# para a vizinhanca da banda sem que os pontos se toquem.
ZOOM_NM = 30

# Paleta categorica do projeto, validada em OKLab sob visao normal e as tres
# dicromacias (ver o bloco de cores de figuraResumo/figura_resumo.py). Dentro
# de cada painel ha so duas curvas, e elas tambem se separam por estilo:
# a "antes" e uma linha fina continua, a "depois" e uma linha com marcadores.
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}

COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"
COR_NEUTRA = "#b8b6b0"

REGIOES = [("VIS", 400, 700), ("RE", 700, 780), ("NIR", 780, 1350),
           ("SWIR1", 1350, 1800), ("SWIR2", 1800, 2451)]


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

def carregar_grupos(genotipo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela banda a banda e tabela de janelas do genotipo."""
    pasta = DADOS / genotipo
    bandas = pd.read_csv(pasta / "grupos_colinearidade.csv", sep=";")
    janelas = pd.read_csv(pasta / "bandas_representativas.csv", sep=";")
    return bandas, janelas


def carregar_top5(genotipo: str, janelas: pd.DataFrame) -> tuple[list[int], bool]:
    """Top 5 do genotipo, ou cinco representantes espalhados de menor q."""
    caminho = SELECAO / genotipo / "top5_bandas.csv"
    if caminho.exists():
        top5 = pd.read_csv(caminho, sep=";").sort_values("posicao")
        return [int(b) for b in top5["banda_nm"]], True

    # O q satura em varias regioes, entao ordenar por ele e cortar os cinco
    # primeiros devolveria cinco janelas vizinhas -- cinco recortes iguais.
    # A distancia minima obriga os recortes a cobrirem regioes diferentes.
    ordenadas = janelas.sort_values("q_condicao_representante")
    escolhidas: list[int] = []
    for banda in ordenadas["banda_representante_nm"].astype(int):
        if all(abs(banda - e) >= 4 * ZOOM_NM for e in escolhidas):
            escolhidas.append(int(banda))
        if len(escolhidas) == 5:
            break
    return escolhidas, False


def erro_de_reconstrucao(
    w: np.ndarray,
    media: np.ndarray,
    bandas_rep: np.ndarray,
) -> tuple[float, float]:
    """Desvio maximo e RMSE ao reconstruir a curva so com os representantes."""
    mask = np.isin(w.astype(int), bandas_rep)
    reconstruida = np.interp(w, w[mask], media[mask])
    residuo = media - reconstruida
    return float(np.abs(residuo).max()), float(np.sqrt((residuo**2).mean()))


# --------------------------------------------------------------------------
# Paineis
# --------------------------------------------------------------------------

def anotar_regioes(ax: plt.Axes) -> None:
    """Sombreia e nomeia as regioes espectrais."""
    for i, (nome, ini, fim) in enumerate(REGIOES):
        if i % 2:
            ax.axvspan(ini, fim, color=COR_NEUTRA, alpha=0.12, zorder=0)
        ax.annotate(nome, xy=((ini + fim) / 2, 0.015),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=7.5, color=COR_TEXTO_FRACO,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.75))


def painel_geral(
    ax: plt.Axes,
    genotipo: str,
    w: np.ndarray,
    media: np.ndarray,
    janelas: pd.DataFrame,
    top5: list[int],
    n_amostras: int,
) -> None:
    """Espectro antes e depois da reducao, no eixo inteiro."""
    cor = COR_GENOTIPO[genotipo]
    bandas_rep = janelas["banda_representante_nm"].astype(int).to_numpy()
    mask = np.isin(w.astype(int), bandas_rep)

    # A curva "antes" e uma faixa larga e clara; a "depois" e uma linha fina
    # por cima. As duas praticamente coincidem -- e esse e o resultado -- entao
    # o contraste de espessura e o que deixa ver que sao duas, com o halo cinza
    # aparecendo onde a reconstrucao se afasta do original.
    ax.plot(w, media, color=COR_NEUTRA, linewidth=4.5, solid_capstyle="round",
            zorder=2, label=f"Antes: {len(w)} bandas")
    ax.plot(w[mask], media[mask], color=cor, linewidth=1.1, marker="o",
            markersize=2.4, markerfacecolor=cor, markeredgecolor=cor,
            zorder=3, label=f"Depois: {int(mask.sum())} representantes")

    # Tres das cinco bandas caem dentro de 30 nm em todos os genotipos: os
    # rotulos alternam de altura para nao se sobreporem.
    for i, banda in enumerate(sorted(top5)):
        ax.axvline(banda, color=COR_TEXTO, linestyle=":", linewidth=1,
                   alpha=0.65, zorder=1)
        ax.annotate(f"{banda}", xy=(banda, 0.985 - 0.11 * (i % 2)),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=7.5, fontweight="bold", color=COR_TEXTO,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.8))

    anotar_regioes(ax)

    desvio, rmse = erro_de_reconstrucao(w, media, bandas_rep)
    reducao = 1 - len(janelas) / len(w)
    ax.set_xlim(w.min(), w.max())
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("Media do espectro (SNV)", fontsize=9)
    ax.set_title(
        f"{genotipo}  -  {n_amostras} amostras  |  {len(w)} bandas -> "
        f"{len(janelas)} janelas de ate 10 nm ({reducao:.0%})  |  "
        f"reconstruindo a curva so com os representantes: desvio maximo "
        f"{desvio:.3f} SNV, RMSE {rmse:.3f}",
        fontsize=10.5, fontweight="bold", loc="left", color=COR_TEXTO,
    )
    ax.grid(True, alpha=0.3)


def painel_zoom(
    ax: plt.Axes,
    genotipo: str,
    w: np.ndarray,
    media: np.ndarray,
    bandas: pd.DataFrame,
    banda_alvo: int,
    posicao: int,
) -> None:
    """Recorte de 60 nm em torno de uma banda, com as janelas visiveis."""
    cor = COR_GENOTIPO[genotipo]
    # Nas pontas do espectro o recorte sai da grade; encostar no limite evita
    # deixar meio painel vazio em bandas como 404 nm.
    ini = max(banda_alvo - ZOOM_NM, float(w.min()))
    fim = min(banda_alvo + ZOOM_NM, float(w.max()))

    recorte = bandas[bandas["banda_nm"].between(ini, fim)]
    mask = (w >= ini) & (w <= fim)

    # Faixas alternadas: cada janela de 10 nm e um bloco de cor, e a
    # alternancia e o que deixa a fronteira entre elas legivel.
    for i, (_, sub) in enumerate(recorte.groupby("grupo", sort=True)):
        if i % 2:
            ax.axvspan(sub["banda_nm"].min() - 0.5, sub["banda_nm"].max() + 0.5,
                       color=COR_NEUTRA, alpha=0.22, linewidth=0, zorder=0)

    ax.plot(w[mask], media[mask], color=COR_NEUTRA, linewidth=2.6,
            solid_capstyle="round", zorder=2)
    ax.scatter(w[mask], media[mask], s=11, color=COR_TEXTO_FRACO, alpha=0.75,
               zorder=3, linewidths=0)

    reps = recorte[recorte["representante"]]["banda_nm"].astype(int).to_numpy()
    mask_rep = np.isin(w.astype(int), reps) & mask
    ax.scatter(w[mask_rep], media[mask_rep], s=34, color=cor, zorder=4,
               edgecolors="white", linewidths=0.8)

    # A banda da Top 5 e sempre um representante -- a selecao so escolhe entre
    # eles. O anel a destaca das outras cinco janelas do recorte.
    alvo = w.astype(int) == banda_alvo
    ax.scatter(w[alvo], media[alvo], s=110, facecolors="none",
               edgecolors=COR_TEXTO, linewidths=1.4, zorder=5)
    ax.axvline(banda_alvo, color=COR_TEXTO, linestyle=":", linewidth=1,
               alpha=0.6, zorder=1)

    ax.set_xlim(ini, fim)
    ax.set_title(f"{posicao}. {banda_alvo} nm", fontsize=9, fontweight="bold",
                 color=COR_TEXTO)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.25)


# --------------------------------------------------------------------------
# Figura
# --------------------------------------------------------------------------

def gerar(meta: pd.DataFrame, espectro: np.ndarray, w: np.ndarray) -> None:
    """Monta a figura inteira: dois blocos de linha por genotipo."""
    fig = plt.figure(figsize=(17.5, 16.5))
    gs = fig.add_gridspec(
        6, 5,
        height_ratios=[1.35, 0.85] * 3,
        hspace=0.55, wspace=0.28,
        top=0.925, bottom=0.055, left=0.055, right=0.985,
    )

    houve_fallback = False
    for i, genotipo in enumerate(GENOTIPOS):
        mask = (meta["genotipo"] == genotipo).to_numpy()
        media = espectro[mask].mean(axis=0)

        bandas, janelas = carregar_grupos(genotipo)
        top5, tem_selecao = carregar_top5(genotipo, janelas)
        houve_fallback |= not tem_selecao

        painel_geral(fig.add_subplot(gs[2 * i, :]), genotipo, w, media,
                     janelas, top5, int(mask.sum()))

        for j, banda in enumerate(top5):
            painel_zoom(fig.add_subplot(gs[2 * i + 1, j]), genotipo, w, media,
                        bandas, banda, j + 1)

    fonte = ("Top 5 de selecaoVariaveis" if not houve_fallback
             else "cinco representantes de menor q (Top 5 indisponivel)")
    fig.suptitle(
        "Reducao de colinearidade: o espectro antes e depois, por genotipo\n"
        f"Espectro normalizado (SNV), turno da manha  -  recortes de "
        f"{2 * ZOOM_NM} nm centrados na {fonte}",
        fontsize=15, fontweight="bold", y=0.982, color=COR_TEXTO,
    )

    fig.legend(
        handles=[
            Line2D([0], [0], color=COR_NEUTRA, linewidth=4.5,
                   label="Antes: as 2051 bandas"),
            Line2D([0], [0], color=COR_TEXTO_FRACO, linewidth=0, marker="o",
                   markersize=4, label="Cada uma das 10 bandas da janela"),
            Line2D([0], [0], color=COR_TEXTO_FRACO, linewidth=0, marker="s",
                   markersize=9, alpha=0.35, label="Janela de 10 nm"),
            Line2D([0], [0], color="none", marker="o", markersize=8,
                   markerfacecolor=COR_NEUTRA, markeredgecolor="white",
                   label="Depois: o representante da janela"),
            Line2D([0], [0], color="none", marker="o", markersize=10,
                   markerfacecolor="none", markeredgecolor=COR_TEXTO,
                   label="Banda da Top 5"),
        ],
        loc="lower center", ncol=5, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.017),
    )

    fig.text(
        0.055, 0.004,
        "Os representantes usam a cor do genotipo. Cada janela cobre no maximo "
        "10 nm e so agrupa bandas com |Spearman| > 0.80 contra todas as ja "
        "aceitas; dentro dela, o representante e a banda de menor q para o "
        "efeito de condicao naquele genotipo.",
        fontsize=9, color=COR_TEXTO_FRACO, ha="left", va="bottom",
    )

    fig.savefig(SAIDA, dpi=180)
    print(f"Figura salva em: {SAIDA}")


def main() -> None:
    print("Gerando a figura da reducao de colinearidade...\n")

    meta, espectro, w = carregar(ESTAGIO, turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")

    for genotipo in GENOTIPOS:
        _, janelas = carregar_grupos(genotipo)
        top5, tem_selecao = carregar_top5(genotipo, janelas)
        origem = "Top 5" if tem_selecao else "menor q (fallback)"
        print(f"  {genotipo}: {len(janelas)} janelas  |  {origem}: "
              f"{', '.join(str(b) for b in top5)}")

    print()
    gerar(meta, espectro, w)
    print("Concluido.")


if __name__ == "__main__":
    main()
