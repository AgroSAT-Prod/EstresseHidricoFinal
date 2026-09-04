#!/usr/bin/env python3
"""Curvas BR16 por dia com as bandas VIP do PLS-DA apos Spearman.

Mostra as medias espectrais de IRRIG e NIRRIG (leituras da manha) e marca as
15 bandas com maior VIP do PLS-DA de cada dia. As cores separam as faixas de
ranking Top 5, Top 10 (posicoes 6--10) e Top 15 (posicoes 11--15).
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

SAIDA = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "figuras"
DIAS = ("D02", "D03", "D04", "D05", "D06", "D09", "D10")
CORES_CURVAS = {"IRRIG": "#2878B5", "NIRRIG": "#C63D4B"}
CORES_VIP = {"Top 5": "#7A1FA2", "Top 10 (6-10)": "#E67E22", "Top 15 (11-15)": "#169B62"}


def faixa_vip(posicao: int) -> str:
    if posicao <= 5:
        return "Top 5"
    if posicao <= 10:
        return "Top 10 (6-10)"
    return "Top 15 (11-15)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvas BR16 com bandas VIP por limiar de Spearman.")
    parser.add_argument("--p-limiar", type=float, default=0.0001)
    parser.add_argument("--genotipo", choices=("BR16", "CD202", "EMB48"), default="BR16")
    parser.add_argument("--entrada", type=Path, default=None)
    args = parser.parse_args()
    if not 0 < args.p_limiar < 1:
        parser.error("--p-limiar deve estar entre 0 e 1.")
    codigo_p = f"p{args.p_limiar:.4g}".replace(".", "")
    entrada = args.entrada or (ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / f"plsda_vip_spearman_{codigo_p}_5seeds_kfold4" / args.genotipo)
    SAIDA.mkdir(parents=True, exist_ok=True)
    meta, X, ondas = carregar("normalizado", turno="manha")
    indice_onda = {int(onda): indice for indice, onda in enumerate(ondas.astype(int))}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, eixos = plt.subplots(4, 2, figsize=(18, 17), sharex=True, sharey=True)
    eixos = eixos.ravel()

    for eixo, dia in zip(eixos, DIAS):
        mascara = (meta.genotipo.eq(args.genotipo) & meta.dia.eq(dia)).to_numpy()
        metad, Xd = meta.loc[mascara].reset_index(drop=True), X[mascara]
        medias = {
            condicao: Xd[metad.condicao.eq(condicao).to_numpy()].mean(axis=0)
            for condicao in CORES_CURVAS
        }
        for condicao, cor in CORES_CURVAS.items():
            eixo.plot(ondas, medias[condicao], color=cor, lw=1.35, label=condicao, zorder=2)

        ranking = pd.read_csv(entrada / dia / "vip_bandas.csv", sep=";").head(15)
        for linha in ranking.itertuples(index=False):
            banda, faixa = int(linha.banda_nm), faixa_vip(int(linha.posicao))
            indice, cor = indice_onda[banda], CORES_VIP[faixa]
            eixo.axvline(banda, color=cor, lw=1.0, alpha=.65, zorder=1)
            for media in medias.values():
                eixo.scatter(banda, media[indice], s=25, color=cor, edgecolor="white", linewidth=.45, zorder=3)

        top5 = ", ".join(str(int(x)) for x in ranking.head(5).banda_nm)
        eixo.text(.015, .025, f"Top 5: {top5} nm", transform=eixo.transAxes, fontsize=8.3,
                  color=CORES_VIP["Top 5"], weight="bold",
                  bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "edgecolor": "#C9C9C9", "alpha": .9})
        eixo.set_title(dia, loc="left", fontweight="bold")
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.grid(axis="x", linestyle=":", alpha=.35)

    eixos[-1].axis("off")
    legenda = [
        Line2D([0], [0], color=CORES_CURVAS["IRRIG"], lw=2, label="IRRIG (média)"),
        Line2D([0], [0], color=CORES_CURVAS["NIRRIG"], lw=2, label="NIRRIG (média)"),
        *[Line2D([0], [0], color=cor, lw=2, label=faixa) for faixa, cor in CORES_VIP.items()],
    ]
    eixos[-1].legend(handles=legenda, loc="center", frameon=True, fontsize=12, title="Curvas e ranking VIP")
    eixos[-1].text(.5, .33, "Marcadores e linhas verticais indicam as\nfaixas do ranking VIP por dia.", ha="center", va="center", fontsize=11)

    for eixo in eixos[:-1]:
        eixo.set_xlim(float(ondas.min()), float(ondas.max()))
        eixo.set_ylabel("Reflectância SNV")
    for eixo in eixos[6:]:
        eixo.set_xlabel("Comprimento de onda (nm)")

    fig.suptitle(f"{args.genotipo}: curvas espectrais e detecção de bandas VIP por dia", fontsize=18, fontweight="bold", y=.985)
    fig.text(.5, .958, f"PLS-DA: IRRIG × NIRRIG | Spearman p < {args.p_limiar:.4g} | janela de 10 nm | dados da manhã", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extensao in ("png", "pdf"):
        fig.savefig(SAIDA / f"curvas_{args.genotipo}_VIP_top5_top10_top15_spearman_{codigo_p}.{extensao}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figuras salvas em {SAIDA.resolve()}")


if __name__ == "__main__":
    main()
