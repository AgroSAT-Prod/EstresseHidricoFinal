#!/usr/bin/env python3
"""Curvas completas e representantes de um cenario do experimento Spearman."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402


RESULTADOS_PADRAO = (
    ROOT / "selecaoVariaveis" / "dataset_gerado"
    / "experimento_spearman_por_dia"
)


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plota um cenario ja calculado do experimento multilimiar."
    )
    parser.add_argument(
        "--genotipo", default="BR16", choices=("BR16", "CD202", "EMB48")
    )
    parser.add_argument("--dia", default="D02")
    parser.add_argument(
        "--correlacao-maxima", "--limiar", dest="correlacao_maxima",
        type=float, default=0.80,
        help="Correlacao absoluta maxima aceita entre as bandas do Top.",
    )
    parser.add_argument("--resultados", type=Path, default=RESULTADOS_PADRAO)
    parser.add_argument(
        "--saida", type=Path,
        help="PNG de saida; por padrao usa nome derivado do cenario.",
    )
    return parser


def main() -> None:
    args = criar_parser().parse_args()
    detalhe_path = args.resultados / "bandas_detalhadas.csv"
    top_path = args.resultados / "top5_bandas.csv"
    for caminho in (detalhe_path, top_path):
        if not caminho.exists():
            raise SystemExit(f"Resultado nao encontrado: {caminho}")

    detalhe = pd.read_csv(detalhe_path, sep=";")
    top5 = pd.read_csv(top_path, sep=";")
    filtro_detalhe = (
        detalhe["genotipo"].eq(args.genotipo)
        & detalhe["dia"].eq(args.dia)
        & np.isclose(
            detalhe["correlacao_maxima_aceita_top"], args.correlacao_maxima
        )
    )
    filtro_top = (
        top5["genotipo"].eq(args.genotipo)
        & top5["dia"].eq(args.dia)
        & np.isclose(
            top5["correlacao_maxima_aceita_top"], args.correlacao_maxima
        )
    )
    detalhe = detalhe.loc[filtro_detalhe].sort_values("banda_nm")
    top5 = top5.loc[filtro_top].sort_values("posicao")
    if detalhe.empty:
        raise SystemExit(
            f"Cenario ausente: {args.genotipo}/{args.dia}/"
            f"Top |rho|<{args.correlacao_maxima:.2f}"
        )

    meta, X, w = carregar("normalizado", turno="manha")
    mask = (
        meta["genotipo"].eq(args.genotipo) & meta["dia"].eq(args.dia)
    ).to_numpy()
    meta, X = meta.loc[mask].reset_index(drop=True), X[mask]
    if meta.empty:
        raise SystemExit(f"Sem amostras para {args.genotipo}/{args.dia}.")
    condicao = meta["condicao"].to_numpy()
    medias = {
        nome: X[condicao == nome].mean(axis=0)
        for nome in ("IRRIG", "NIRRIG")
    }

    mapa_indices = {int(banda): i for i, banda in enumerate(w)}
    bandas_representantes = detalhe.loc[detalhe["representante"], "banda_nm"].astype(int)
    indices_representantes = np.array(
        [mapa_indices[banda] for banda in bandas_representantes], dtype=int
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 9), sharey=True, constrained_layout=True
    )
    cores = {"IRRIG": "#2878B5", "NIRRIG": "#C63D4B"}

    for nome, media in medias.items():
        axes[0].plot(w, media, color=cores[nome], lw=1.4, label=nome)
    axes[0].set_title(
        f"{args.genotipo} — {args.dia}, manha: curvas medias completas "
        f"({len(w):,} bandas)".replace(",", "."),
        loc="left", weight="bold",
    )
    axes[0].legend(title="Condicao", ncol=2, frameon=True)

    for nome, media in medias.items():
        axes[1].plot(
            w[indices_representantes], media[indices_representantes],
            color=cores[nome], lw=0.8, alpha=0.55,
        )
        axes[1].scatter(
            w[indices_representantes], media[indices_representantes],
            color=cores[nome], s=10, alpha=0.75,
            label=f"{nome} — representantes",
        )
    for banda in top5["banda_nm"].astype(int):
        idx = mapa_indices[banda]
        axes[1].axvline(banda, color="#E69F00", lw=1.0, alpha=0.8, zorder=1)
        for media in medias.values():
            axes[1].scatter(
                banda, media[idx], s=78, marker="o", color="#E69F00",
                edgecolor="black", linewidth=0.7, zorder=5,
            )
    axes[1].set_title(
        f"Grupos com |rho| > {args.correlacao_maxima:.2f} e janela < 10 nm: "
        f"{len(indices_representantes)} representantes; Top PLS-DA com "
        f"|rho| < {args.correlacao_maxima:.2f} em amarelo",
        loc="left", weight="bold",
    )
    bandas_texto = ", ".join(str(v) for v in top5["banda_nm"].astype(int))
    axes[1].text(
        0.015, 0.06,
        "Top 5 PLS-DA (nm): " + (bandas_texto or "nenhuma"),
        transform=axes[1].transAxes, fontsize=9, color="#805400", weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF4CC", "edgecolor": "#E69F00"},
    )

    for eixo in axes:
        eixo.set_xlim(w.min(), w.max())
        eixo.set_ylabel("Reflectancia SNV")
        eixo.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Comprimento de onda (nm)")

    saida = args.saida or (
        ROOT / "analiseBR16PorDia"
        / f"curvas_{args.genotipo}_{args.dia}_spearman_"
        f"{args.correlacao_maxima:.2f}.png"
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em {saida.resolve()}")


if __name__ == "__main__":
    main()
