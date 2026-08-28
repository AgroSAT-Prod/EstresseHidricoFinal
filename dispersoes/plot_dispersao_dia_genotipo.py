#!/usr/bin/env python3
"""Dispersao espectral dia a dia, uma figura por genotipo.

Onde `plot_curvas_genotipo.py` joga todas as leituras num painel so, aqui a
nuvem vira estatistica e o dia abre em subplots. Cada subplot resume as
leituras daquele genotipo naquele dia, em cada condicao, em tres niveis:

- linha grossa continua: media espectral
- linha tracejada (traco espacado): intervalo de confianca de 95% da media
- linha pontilhada fina: minimo e maximo observados banda a banda

Os tres juntos separam duas coisas que a nuvem crua confunde: o IC diz o quao
bem a media esta determinada (com n=32 por dia e condicao ele e estreito), e o
envelope min/max diz o quanto as leituras individuais variam. Um IC apertado
dentro de um envelope largo -- que e o caso -- significa media confiavel sobre
uma populacao heterogenea, e nao leituras homogeneas.

IRRIG e NIRRIG saem separados, na mesma cor que o resto do projeto usa. A
leitura util e onde os dois ICs deixam de se cruzar: e ali que a diferenca
entre condicoes passa a ser maior que a incerteza da propria media.

Grid 7x2 por figura -- 7 linhas (D02 a D10, os dias com coleta de manha) e 2
colunas (recortado, normalizado com SNV). Uma figura por genotipo.

Uso:
    python plot_dispersao_dia_genotipo.py
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
from plot_preprocessamento import COR_IRRIG, COR_NIRRIG  # noqa: E402

TURNO = "manha"

CONDICOES = ["IRRIG", "NIRRIG"]
COR_CONDICAO = {"IRRIG": COR_IRRIG, "NIRRIG": COR_NIRRIG}
CONFIANCA = 0.95


def resumo_banda(espectro: np.ndarray) -> dict[str, np.ndarray]:
    """Media, limites do IC de 95% da media, minimo e maximo por banda."""
    n = espectro.shape[0]
    media = espectro.mean(axis=0)
    minimo = espectro.min(axis=0)
    maximo = espectro.max(axis=0)

    # IC da media, nao da observacao: erro padrao com t de Student. Com n=64 o
    # t esta praticamente em 1.96, mas usar o t deixa o codigo correto tambem
    # nos dias em que o n cair.
    if n > 1:
        erro = espectro.std(axis=0, ddof=1) / np.sqrt(n)
        margem = t_dist.ppf(0.5 + CONFIANCA / 2, df=n - 1) * erro
    else:
        margem = np.zeros_like(media)

    return {
        "media": media,
        "ic_inf": media - margem,
        "ic_sup": media + margem,
        "min": minimo,
        "max": maximo,
        "n": n,
    }


def plot_celula(
    ax: plt.Axes,
    w: np.ndarray,
    resumo: dict[str, np.ndarray],
    titulo: str,
    unidade: str,
) -> None:
    """Um subplot: media, IC de 95% e envelope min/max das duas condicoes."""
    # Todos os envelopes antes de qualquer media: assim nenhuma das duas
    # medias fica escondida atras do min/max da outra condicao.
    for condicao in CONDICOES:
        if condicao not in resumo:
            continue
        cor = COR_CONDICAO[condicao]
        r = resumo[condicao]
        ax.plot(w, r["min"], color=cor, linewidth=0.7, linestyle=(0, (1, 2)))
        ax.plot(w, r["max"], color=cor, linewidth=0.7, linestyle=(0, (1, 2)))
        ax.plot(w, r["ic_inf"], color=cor, linewidth=1.1, linestyle=(0, (6, 4)))
        ax.plot(w, r["ic_sup"], color=cor, linewidth=1.1, linestyle=(0, (6, 4)))

    for condicao in CONDICOES:
        if condicao not in resumo:
            continue
        ax.plot(w, resumo[condicao]["media"], color=COR_CONDICAO[condicao],
                linewidth=2.2, solid_capstyle="round",
                # Halo curto de proposito: o IC corre colado na media, e um
                # contorno largo apagaria justamente a linha que ele delimita.
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                              pe.Normal()])

    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    ax.set_ylabel(unidade, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def legenda_figura(fig: plt.Figure) -> None:
    """Legenda em dois eixos: cor = condicao, traco = estatistica.

    Seis linhas na legenda seriam so o produto cartesiano das duas coisas; o
    par de chaves separado e mais curto de ler e nao cresce se entrar uma
    terceira condicao.
    """
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=COR_CONDICAO[c], linewidth=2.2, label=c)
        for c in CONDICOES
    ]
    handles += [
        Line2D([], [], color="#555555", linewidth=2.2, label="Media"),
        Line2D([], [], color="#555555", linewidth=1.1, linestyle=(0, (6, 4)),
               label="IC 95% da media"),
        Line2D([], [], color="#555555", linewidth=0.7, linestyle=(0, (1, 2)),
               label="Minimo / maximo"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10)


def gerar_figura(
    genotipo: str,
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
    dias: list[str],
) -> Path:
    """Gera e salva a figura de um genotipo."""
    do_genotipo = (meta["genotipo"] == genotipo).to_numpy()

    resumos: dict[tuple[str, str], dict[str, dict]] = {}
    for chave, _, _ in ESTAGIOS:
        for dia in dias:
            do_dia = do_genotipo & (meta["dia"] == dia).to_numpy()
            por_condicao = {}
            for condicao in CONDICOES:
                mask = do_dia & (meta["condicao"] == condicao).to_numpy()
                if mask.any():
                    por_condicao[condicao] = resumo_banda(estagios[chave][mask])
            if por_condicao:
                resumos[(chave, dia)] = por_condicao

    # Escala de y compartilhada por estagio: sem isso cada dia escolhe seu
    # proprio limite e a comparacao entre dias -- o ponto do grid -- some.
    limites = {}
    for chave, _, _ in ESTAGIOS:
        valores = np.concatenate(
            [r[k] for (c, _), por_condicao in resumos.items() if c == chave
             for r in por_condicao.values() for k in ("min", "max")]
        )
        folga = 0.05 * (valores.max() - valores.min())
        limites[chave] = (valores.min() - folga, valores.max() + folga)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(dias), len(ESTAGIOS),
                             figsize=(7.5 * len(ESTAGIOS), 3.4 * len(dias)))

    for i, dia in enumerate(dias):
        for j, (chave, rotulo, unidade) in enumerate(ESTAGIOS):
            ax = axes[i, j]
            resumo = resumos.get((chave, dia))
            if resumo is None:
                ax.set_axis_off()
                continue
            ns = ", ".join(f"{c} n={resumo[c]['n']}" for c in CONDICOES
                           if c in resumo)
            plot_celula(ax, w, resumo, f"{dia} - {rotulo}  ({ns})", unidade)
            ax.set_ylim(*limites[chave])

    legenda_figura(fig)

    fig.suptitle(
        f"Genotipo {genotipo} - dispersao espectral dia a dia\n"
        "Turno da manha, IRRIG e NIRRIG separados  -  escala do eixo y "
        "compartilhada por estagio",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.975])
    saida = (ROOT.parent / "dispersoes") / f"dispersao_dia_{genotipo}.png"
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saida


def main() -> None:
    print("Gerando figuras de dispersao espectral por genotipo...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")

    dias = sorted(meta["dia"].unique())
    genotipos = sorted(meta["genotipo"].unique())
    print(f"  Dias: {', '.join(dias)}")
    print(f"  Genotipos: {', '.join(genotipos)}\n")

    for genotipo in genotipos:
        saida = gerar_figura(genotipo, meta, estagios, w, dias)
        print(f"  {genotipo}: {saida.name}")

    print("\nConcluido.")


if __name__ == "__main__":
    main()
