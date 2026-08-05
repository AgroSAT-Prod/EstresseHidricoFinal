#!/usr/bin/env python3
"""Media temporal dos dias para um genotipo, com IC de 95% em hachura.

Mesma leitura de `plot_dispersao_dia_genotipo.py`, com o dia fechado em vez de
aberto: em vez de um subplot por dia, cada dia vira uma observacao e o painel
mostra a media temporal do periodo. Um painel por estagio (recortado e
normalizado), IRRIG e NIRRIG separados por cor.

A unidade estatistica aqui e o dia, nao a leitura. Primeiro cada dia e
reduzido a sua media espectral; depois media, IC e envelope sao calculados
sobre esses 7 valores diarios. Isso responde "onde a diferenca entre condicoes
se sustenta ao longo do experimento" -- um IC calculado sobre as ~450 leituras
individuais responderia outra coisa, e seria estreito demais por tratar
leituras do mesmo dia como independentes.

Os tres niveis de cada condicao:

- linha grossa continua: media dos dias
- faixa hachurada: IC de 95% da media (t de Student, n = numero de dias)
- linha pontilhada fina: dia de menor e de maior media, banda a banda

Uma figura por genotipo, colorida. O genotipo de referencia (`GENOTIPO_CINZA`)
ganha tambem uma versao em escala de cinza, para impressao em preto e branco.

Uso:
    python plot_media_temporal_genotipo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402
from plot_curvas_genotipo import ESTAGIOS  # noqa: E402
from plot_dispersao_dia_genotipo import CONDICOES, COR_CONDICAO  # noqa: E402

GENOTIPO_CINZA = "BR16"
TURNO = "manha"
CONFIANCA = 0.95

# Hachuras opostas: onde as duas faixas se cruzam, mesma direcao viraria uma
# textura so e a sobreposicao ficaria ilegivel.
HACHURA = {"IRRIG": "///", "NIRRIG": "\\\\\\"}

# Versao em escala de cinza (impressao em preto e branco). Os dois tons estao
# separados por ~45 pontos de luminancia -- em cinza a hachura sozinha nao
# aguenta a identificacao, entao o valor do tom precisa carregar parte dela.
CINZA_CONDICAO = {"IRRIG": "#262626", "NIRRIG": "#949494"}


def resumo_temporal(
    espectro: np.ndarray,
    dias_amostra: np.ndarray,
    dias: list[str],
) -> dict:
    """Media, IC de 95% e envelope calculados sobre as medias diarias."""
    medias_dia = np.array([
        espectro[dias_amostra == dia].mean(axis=0)
        for dia in dias if (dias_amostra == dia).any()
    ])
    n = medias_dia.shape[0]

    media = medias_dia.mean(axis=0)
    if n > 1:
        erro = medias_dia.std(axis=0, ddof=1) / np.sqrt(n)
        margem = t_dist.ppf(0.5 + CONFIANCA / 2, df=n - 1) * erro
    else:
        margem = np.zeros_like(media)

    return {
        "media": media,
        "ic_inf": media - margem,
        "ic_sup": media + margem,
        "min": medias_dia.min(axis=0),
        "max": medias_dia.max(axis=0),
        "n": n,
    }


def plot_painel(
    ax: plt.Axes,
    w: np.ndarray,
    resumo: dict[str, dict],
    cores: dict[str, str],
    titulo: str,
    unidade: str,
) -> None:
    """Um estagio: media temporal, IC hachurado e envelope das duas condicoes."""
    for condicao in CONDICOES:
        if condicao not in resumo:
            continue
        cor = cores[condicao]
        r = resumo[condicao]
        # Hachura sem preenchimento: uma faixa solida esconderia a faixa da
        # outra condicao justamente nos trechos em que elas se sobrepoem, que
        # sao os trechos que interessam.
        ax.fill_between(w, r["ic_inf"], r["ic_sup"], facecolor="none",
                        edgecolor=cor, hatch=HACHURA[condicao], linewidth=0.0)
        ax.plot(w, r["ic_inf"], color=cor, linewidth=0.8)
        ax.plot(w, r["ic_sup"], color=cor, linewidth=0.8)
        ax.plot(w, r["min"], color=cor, linewidth=0.7, linestyle=(0, (1, 2)))
        ax.plot(w, r["max"], color=cor, linewidth=0.7, linestyle=(0, (1, 2)))

    for condicao in CONDICOES:
        if condicao not in resumo:
            continue
        ax.plot(w, resumo[condicao]["media"], color=cores[condicao],
                linewidth=2.2, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                              pe.Normal()])

    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel(unidade, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def legenda_figura(fig: plt.Figure, cores: dict[str, str]) -> None:
    """Cor = condicao; traco/hachura = estatistica."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    neutro = "#555555"
    handles = [
        Line2D([], [], color=cores[c], linewidth=2.2, label=c)
        for c in CONDICOES
    ]
    handles += [
        Line2D([], [], color=neutro, linewidth=2.2, label="Media dos dias"),
        Patch(facecolor="none", edgecolor=neutro, hatch="///",
              label="IC 95% da media"),
        Line2D([], [], color=neutro, linewidth=0.7, linestyle=(0, (1, 2)),
               label="Dia minimo / maximo"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10)


def gerar_figura(
    genotipo: str,
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
    dias: list[str],
    cinza: bool = False,
) -> Path:
    """Gera e salva a figura de media temporal do genotipo."""
    cores = CINZA_CONDICAO if cinza else COR_CONDICAO
    do_genotipo = (meta["genotipo"] == genotipo).to_numpy()
    dias_amostra = meta["dia"].to_numpy()

    resumos = {}
    for chave, _, _ in ESTAGIOS:
        por_condicao = {}
        for condicao in CONDICOES:
            mask = do_genotipo & (meta["condicao"] == condicao).to_numpy()
            if mask.any():
                por_condicao[condicao] = resumo_temporal(
                    estagios[chave][mask], dias_amostra[mask], dias
                )
        resumos[chave] = por_condicao

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(ESTAGIOS), figsize=(7.5 * len(ESTAGIOS), 5.5))

    for ax, (chave, rotulo, unidade) in zip(axes, ESTAGIOS):
        plot_painel(ax, w, resumos[chave], cores, rotulo, unidade)

    legenda_figura(fig, cores)

    n_dias = max(r["n"] for r in resumos[ESTAGIOS[0][0]].values())
    fig.suptitle(
        f"Genotipo {genotipo} - media temporal dos dias\n"
        f"Turno da manha, IRRIG e NIRRIG separados  -  cada dia entra como uma "
        f"observacao (n={n_dias} dias: {', '.join(dias)})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.92])
    sufixo = "_cinza" if cinza else ""
    saida = ROOT / f"media_temporal_{genotipo}{sufixo}.png"
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saida


def main() -> None:
    print("Gerando figuras de media temporal...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")

    dias = sorted(meta["dia"].unique())
    genotipos = sorted(meta["genotipo"].unique())
    print(f"  Dias: {', '.join(dias)}")
    print(f"  Genotipos: {', '.join(genotipos)}\n")

    for genotipo in genotipos:
        # A versao em cinza sai apenas para o genotipo de referencia -- e a
        # figura destinada a impressao, e as outras duas servem de apoio.
        modos = (False, True) if genotipo == GENOTIPO_CINZA else (False,)
        for cinza in modos:
            saida = gerar_figura(genotipo, meta, estagios, w, dias, cinza=cinza)
            print(f"  {genotipo} ({'cinza' if cinza else 'cor'}): {saida.name}")

    print("\nConcluido.")


if __name__ == "__main__":
    main()
