#!/usr/bin/env python3
"""Comportamento das cinco bandas otimas nos dados, uma linha por banda.

`selecao_variaveis.py` decide QUAIS bandas sobrevivem ao funil; esta figura
mostra o que elas fazem. Uma linha por banda da Top 5, na ordem do ranking, e
quatro recortes por linha:

1. Distribuicao IRRIG vs NIRRIG com os sete dias juntos -- a separacao que o
   classificador enxerga se ignorar o tempo.
2. Trajetoria ao longo dos sete dias, uma curva por condicao. E aqui que a
   banda mostra se a separacao existe desde o inicio ou se abre com o deficit.
3. As seis celulas genotipo x condicao -- se a banda responde ao estresse nos
   tres materiais ou so em algum deles.
4. Delta de Cliff do contraste NIRRIG vs IRRIG, dia a dia e por genotipo.
   Positivo: NIRRIG acima de IRRIG.

Unidade amostral
----------------
Cada celula genotipo x condicao x dia tem 32 leituras, mas so 4 blocos de
campo -- as 8 leituras de um bloco sao subamostras da mesma parcela. Os
intervalos de confianca do painel 2 saem das MEDIAS POR BLOCO (n = 4, t de
Student com 3 graus de liberdade), nao das leituras: um IC calculado sobre as
32 leituras seria oito vezes estreito demais e daria a impressao de uma
separacao mais firme do que o delineamento sustenta.

Os paineis 1 e 3 mostram as leituras porque sao descritivos -- a forma da
distribuicao e o que interessa ali. O delta de Cliff do painel 4 tambem sai
das leituras: ele e uma probabilidade de superioridade, nao um teste, e nao se
infla com o n.

Os valores estao em unidades de SNV (o estagio em que a selecao rodou), entao
o eixo nao e reflectancia -- e o desvio padrao dentro do proprio espectro.

Uso:
    python plot_bandas_otimas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as t_dist

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT.parent / "testeDiferencaSignificativa"))

from shapiro_normalidade import carregar  # noqa: E402
from comparacao_genotipo_condicao import cliff_delta  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"
SAIDA = ROOT / "bandas_otimas.png"

ESTAGIO = "normalizado"
TURNO = "manha"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]

# A selecao roda por genotipo, entao existe um Top 5 por material. A figura
# desenha o de um deles e mostra como os outros dois respondem nessas mesmas
# bandas; CD202 e o de resposta mais forte ao estresse.
GENOTIPO_TOP5 = "CD202"

COR_CONDICAO = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}

COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"
COR_NEUTRA = "#b8b6b0"

REGIOES = [("VIS", 400, 700), ("RE", 700, 780), ("NIR", 780, 1350),
           ("SWIR1", 1350, 1800), ("SWIR2", 1800, 2451)]

# IC de 95% sobre 4 blocos: t de Student com 3 graus de liberdade, nao 1.96.
CONFIANCA = 0.95


def regiao_de(banda: int) -> str:
    """Regiao espectral em que a banda cai."""
    for nome, inicio, fim in REGIOES:
        if inicio <= banda < fim:
            return nome
    return "?"


def medias_por_bloco(
    valores: np.ndarray,
    meta: pd.DataFrame,
    dia: str,
    condicao: str,
) -> np.ndarray:
    """Media de cada bloco de campo numa celula condicao x dia."""
    mask = (meta["dia"] == dia) & (meta["condicao"] == condicao)
    sub = pd.DataFrame({"bloco": meta.loc[mask, "bloco"], "v": valores[mask.to_numpy()]})
    return sub.groupby("bloco")["v"].mean().to_numpy()


def ic_media(amostra: np.ndarray) -> tuple[float, float]:
    """Media e semi-amplitude do IC de 95%, com a t do proprio n."""
    n = len(amostra)
    media = float(np.mean(amostra))
    if n < 2:
        return media, 0.0
    erro = float(np.std(amostra, ddof=1) / np.sqrt(n))
    return media, float(t_dist.ppf(0.5 + CONFIANCA / 2, n - 1) * erro)


def delta_por_dia(
    valores: np.ndarray,
    meta: pd.DataFrame,
    dias: list[str],
    genotipo: str | None = None,
) -> np.ndarray:
    """Delta de Cliff de NIRRIG contra IRRIG em cada dia.

    Os postos sao recalculados dentro de cada dia (e de cada genotipo, quando
    pedido): o contraste e entre as duas condicoes ali, e ordenar junto com os
    outros dias misturaria a deriva temporal no tamanho de efeito.
    """
    saida = np.full(len(dias), np.nan)
    for i, dia in enumerate(dias):
        mask = (meta["dia"] == dia).to_numpy().copy()
        if genotipo is not None:
            mask &= (meta["genotipo"] == genotipo).to_numpy()
        cond = meta.loc[mask, "condicao"].to_numpy()
        idx_n = (cond == "NIRRIG").nonzero()[0]
        idx_i = (cond == "IRRIG").nonzero()[0]
        if len(idx_n) < 2 or len(idx_i) < 2:
            continue
        postos = rankdata(valores[mask].reshape(-1, 1), axis=0)
        saida[i] = cliff_delta(postos, idx_n, idx_i)[0]
    return saida


def estilizar_violino(partes: dict, cor: str) -> None:
    """Corpo na cor da condicao, barras internas discretas."""
    for corpo in partes["bodies"]:
        corpo.set_facecolor(cor)
        corpo.set_alpha(0.35)
        corpo.set_edgecolor(cor)
        corpo.set_linewidth(1.0)
    for chave in ("cmins", "cmaxes", "cbars", "cmedians"):
        if chave in partes:
            partes[chave].set_color(cor)
            partes[chave].set_linewidth(1.2)


def painel_distribuicao(
    ax: plt.Axes,
    valores: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    """Violino + caixa das duas condicoes, com os sete dias juntos."""
    dados = [valores[(meta["condicao"] == c).to_numpy()] for c in CONDICOES]

    for i, (dado, condicao) in enumerate(zip(dados, CONDICOES)):
        partes = ax.violinplot([dado], positions=[i], widths=0.75,
                               showmedians=True, showextrema=True)
        estilizar_violino(partes, COR_CONDICAO[condicao])

    caixas = ax.boxplot(dados, positions=range(len(CONDICOES)), widths=0.16,
                        patch_artist=True, showfliers=False, zorder=3)
    for caixa, condicao in zip(caixas["boxes"], CONDICOES):
        caixa.set_facecolor("white")
        caixa.set_edgecolor(COR_CONDICAO[condicao])
        caixa.set_linewidth(1.1)
    for chave in ("whiskers", "caps", "medians"):
        for artista in caixas[chave]:
            artista.set_color(COR_TEXTO)
            artista.set_linewidth(1.0)

    delta = cliff_delta(
        rankdata(valores.reshape(-1, 1), axis=0),
        (meta["condicao"] == "NIRRIG").to_numpy().nonzero()[0],
        (meta["condicao"] == "IRRIG").to_numpy().nonzero()[0],
    )[0]
    ax.text(0.97, 0.95, f"delta = {delta:+.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, fontweight="bold",
            color=COR_TEXTO_FRACO)

    ax.set_xticks(range(len(CONDICOES)))
    ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(CONDICOES, dados)],
                       fontsize=8)
    ax.set_ylabel("Valor (SNV)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.3, axis="y")


def painel_trajetoria(
    ax: plt.Axes,
    valores: np.ndarray,
    meta: pd.DataFrame,
    dias: list[str],
) -> None:
    """Media por dia e condicao, com IC de 95% sobre as medias de bloco."""
    x = np.arange(len(dias))

    for condicao in CONDICOES:
        medias, margens = [], []
        for dia in dias:
            blocos = medias_por_bloco(valores, meta, dia, condicao)
            media, margem = ic_media(blocos)
            medias.append(media)
            margens.append(margem)
        medias = np.array(medias)
        margens = np.array(margens)

        ax.fill_between(x, medias - margens, medias + margens,
                        color=COR_CONDICAO[condicao], alpha=0.18, linewidth=0)
        ax.plot(x, medias, color=COR_CONDICAO[condicao], linewidth=1.8,
                marker="o", markersize=4, label=condicao)

    ax.set_xticks(x)
    ax.set_xticklabels(dias, fontsize=7.5)
    ax.set_xlim(-0.4, len(dias) - 0.6)
    ax.set_ylabel("Media (SNV)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.3)


def painel_celulas(
    ax: plt.Axes,
    valores: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    """As seis celulas genotipo x condicao, com os sete dias juntos."""
    dados, cores, rotulos = [], [], []
    for genotipo in GENOTIPOS:
        for condicao in CONDICOES:
            mask = ((meta["genotipo"] == genotipo)
                    & (meta["condicao"] == condicao)).to_numpy()
            dados.append(valores[mask])
            cores.append(COR_CONDICAO[condicao])
            rotulos.append(condicao)

    caixas = ax.boxplot(dados, positions=range(len(dados)), widths=0.62,
                        patch_artist=True, showfliers=False)
    for caixa, cor in zip(caixas["boxes"], cores):
        caixa.set_facecolor(cor)
        caixa.set_alpha(0.45)
        caixa.set_edgecolor(cor)
        caixa.set_linewidth(1.0)
    for chave in ("whiskers", "caps", "medians"):
        for artista in caixas[chave]:
            artista.set_color(COR_TEXTO)
            artista.set_linewidth(1.0)

    # Separador entre genotipos, para o olho nao ler as seis caixas como uma serie.
    for corte in range(len(CONDICOES), len(dados), len(CONDICOES)):
        ax.axvline(corte - 0.5, color=COR_NEUTRA, linewidth=0.9, alpha=0.8)

    for i, genotipo in enumerate(GENOTIPOS):
        ax.text(i * len(CONDICOES) + 0.5, 1.02, genotipo,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=COR_GENOTIPO[genotipo])

    ax.set_xticks(range(len(dados)))
    ax.set_xticklabels(rotulos, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("Valor (SNV)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.3, axis="y")


def painel_separacao(
    ax: plt.Axes,
    valores: np.ndarray,
    meta: pd.DataFrame,
    dias: list[str],
) -> None:
    """Delta de Cliff NIRRIG vs IRRIG por dia: barras no geral, linhas por genotipo."""
    x = np.arange(len(dias))
    geral = delta_por_dia(valores, meta, dias)

    ax.bar(x, geral, width=0.62, color=COR_NEUTRA, alpha=0.55, linewidth=0,
           zorder=1, label="Todos os genotipos")

    for genotipo in GENOTIPOS:
        ax.plot(x, delta_por_dia(valores, meta, dias, genotipo),
                color=COR_GENOTIPO[genotipo], linewidth=1.3, marker="o",
                markersize=3.5, zorder=3, label=genotipo)

    ax.axhline(0, color=COR_TEXTO, linewidth=0.9, zorder=2)
    # |delta| = 0.474 e o corte de efeito grande de Romano et al. (2006).
    for sinal in (1, -1):
        ax.axhline(sinal * 0.474, color=COR_TEXTO_FRACO, linestyle=":",
                   linewidth=0.9, alpha=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(dias, fontsize=7.5)
    ax.set_xlim(-0.6, len(dias) - 0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("delta de Cliff", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.3, axis="y")


TITULOS = [
    "1) Distribuicao por condicao\n(sete dias juntos)",
    "2) Trajetoria temporal\n(media +- IC95 dos blocos)",
    "3) Celulas genotipo x condicao\n(sete dias juntos)",
    "4) Separacao por dia\n(delta de Cliff, NIRRIG vs IRRIG)",
]


def gerar(
    meta: pd.DataFrame,
    espectro: np.ndarray,
    w: np.ndarray,
    top5: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    dias = sorted(meta["dia"].unique())
    bandas = [int(b) for b in top5.sort_values("posicao")["banda_nm"]]

    fig, axes = plt.subplots(
        len(bandas), 4, figsize=(21, 22),
        gridspec_kw=dict(hspace=0.32, wspace=0.24,
                         top=0.925, bottom=0.055, left=0.085, right=0.985),
    )

    for i, banda in enumerate(bandas):
        valores = espectro[:, int(np.searchsorted(w, banda))]
        linha = top5[top5["banda_nm"] == banda].iloc[0]

        painel_distribuicao(axes[i, 0], valores, meta)
        painel_trajetoria(axes[i, 1], valores, meta, dias)
        painel_celulas(axes[i, 2], valores, meta)
        painel_separacao(axes[i, 3], valores, meta, dias)

        if i == 0:
            for ax, titulo in zip(axes[i], TITULOS):
                ax.set_title(titulo, fontsize=10, fontweight="bold", pad=26)
        if i == len(bandas) - 1:
            axes[i, 1].set_xlabel("Dia de coleta", fontsize=8)
            axes[i, 3].set_xlabel("Dia de coleta", fontsize=8)

        eixo = axes[i, 0]
        eixo.text(-0.30, 0.62, f"#{int(linha['posicao'])}",
                  transform=eixo.transAxes, ha="center", va="center",
                  fontsize=13, fontweight="bold", color=COR_NEUTRA)
        eixo.text(-0.30, 0.44, f"{banda} nm", transform=eixo.transAxes,
                  ha="center", va="center", fontsize=12, fontweight="bold",
                  color=COR_TEXTO)
        eixo.text(-0.30, 0.30, regiao_de(banda), transform=eixo.transAxes,
                  ha="center", va="center", fontsize=9, color=COR_TEXTO_FRACO)
        eixo.text(-0.30, 0.18, f"VIP {linha['vip']:.2f}",
                  transform=eixo.transAxes, ha="center", va="center",
                  fontsize=8, color=COR_TEXTO_FRACO)

    fig.suptitle(
        f"As cinco bandas otimas de {GENOTIPO_TOP5}: distribuicao, "
        "trajetoria, resposta por genotipo e separacao dia a dia\n"
        "Espectro normalizado (SNV), turno da manha  -  bandas na ordem do "
        "ranking de selecao",
        fontsize=15, fontweight="bold", y=0.975,
    )

    fig.legend(
        handles=[Patch(facecolor=COR_CONDICAO[c], alpha=0.6, label=c)
                 for c in CONDICOES]
        + [Line2D([0], [0], color=COR_GENOTIPO[g], linewidth=2, marker="o",
                  markersize=4, label=g) for g in GENOTIPOS]
        + [Line2D([0], [0], color=COR_TEXTO_FRACO, linestyle=":", linewidth=1.2,
                  label="|delta| = 0.474 (efeito grande)")],
        loc="lower center", ncol=6, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.016),
    )

    fig.text(
        0.085, 0.003,
        "Os IC do painel 2 usam as medias dos 4 blocos de campo como unidade, "
        "nao as 32 leituras da celula. O delta de Cliff e a probabilidade de "
        "uma leitura NIRRIG superar uma IRRIG, menos a probabilidade do "
        "contrario.",
        fontsize=9, color=COR_TEXTO_FRACO, ha="left", va="bottom",
    )

    fig.savefig(SAIDA, dpi=180)
    print(f"Figura salva em: {SAIDA}")


def main() -> None:
    print("Gerando os graficos das bandas otimas...\n")

    top5 = pd.read_csv(SAIDA_DIR / GENOTIPO_TOP5 / "top5_bandas.csv", sep=";")
    meta, espectro, w = carregar(ESTAGIO, turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")

    bandas = [int(b) for b in top5.sort_values("posicao")["banda_nm"]]
    faltando = [b for b in bandas if b not in set(w.astype(int))]
    if faltando:
        raise SystemExit(f"Bandas ausentes no espectro carregado: {faltando}")
    print(f"  Top 5: {', '.join(f'{b} nm ({regiao_de(b)})' for b in bandas)}\n")

    gerar(meta, espectro, w, top5)
    print("Concluido.")


if __name__ == "__main__":
    main()
