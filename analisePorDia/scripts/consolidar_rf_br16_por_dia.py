#!/usr/bin/env python3
"""Consolida RF de BR16 por dia e plota scores e evolução das bandas."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DIAS = ("D02", "D03", "D04", "D05", "D06", "D09", "D10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida RF por dia e genótipo.")
    parser.add_argument("--genotipo", default="BR16", choices=("BR16", "CD202", "EMB48"))
    args = parser.parse_args()
    genotipo = args.genotipo
    base = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "genotipo" / genotipo
    saida = base / "random_forest_por_dia"
    metricas, bandas = [], []
    for dia in DIAS:
        pasta = base / dia / "random_forest"
        resumo = pd.read_csv(pasta / "metricas_resumo.csv", sep=";")
        resumo.insert(0, "dia", dia)
        metricas.append(resumo)

        imp = pd.read_csv(pasta / "importancia_bandas.csv", sep=";")
        # Mantém apenas bandas com q-FDR <= 0,05; empates de permutação são
        # resolvidos por Gini, exatamente como na tabela do RF individual.
        imp = imp[imp["q_fdr"] <= 0.05].copy()
        imp = imp.sort_values(
            ["importancia_permutacao_media", "importancia_gini"], ascending=False
        ).head(5)
        imp.insert(0, "posicao_dia", np.arange(1, len(imp) + 1))
        imp.insert(0, "dia", dia)
        bandas.append(imp)

    scores = pd.concat(metricas, ignore_index=True)
    top5 = pd.concat(bandas, ignore_index=True)
    saida.mkdir(parents=True, exist_ok=True)
    scores.to_csv(saida / "scores_rf_por_dia.csv", sep=";", index=False)
    top5.to_csv(saida / "evolucao_top5_bandas_rf.csv", sep=";", index=False)

    tabela = scores.pivot(index="dia", columns="metrica", values="media").reindex(DIAS)
    fig, (ax_score, ax_band) = plt.subplots(
        2, 1, figsize=(12, 9), constrained_layout=True,
        gridspec_kw={"height_ratios": (1, 1.35)},
    )
    x = np.arange(len(DIAS))
    for metrica, rotulo, cor in [
        ("balanced_accuracy", "Balanced accuracy", "#2878B5"),
        ("auc_roc", "AUC-ROC", "#C63D4B"),
        ("f1", "F1", "#6A9A1F"),
    ]:
        ax_score.plot(x, tabela[metrica], marker="o", lw=2, color=cor, label=rotulo)
    ax_score.axhline(0.5, color="#666666", ls="--", lw=1, label="acaso")
    ax_score.set(xticks=x, xticklabels=DIAS, ylim=(0, 1.05), ylabel="Score")
    ax_score.set_title(f"{genotipo}, manhã — RF por dia: desempenho leave-one-block-out", loc="left", weight="bold")
    ax_score.legend(ncol=4, frameon=True)

    norm = plt.Normalize(top5["importancia_permutacao_media"].min(), top5["importancia_permutacao_media"].max() or 1)
    pontos = ax_band.scatter(
        [DIAS.index(d) for d in top5["dia"]], top5["banda_nm"],
        c=top5["importancia_permutacao_media"], cmap="YlOrRd", norm=norm,
        s=80 + 170 * top5["importancia_gini"] / max(top5["importancia_gini"].max(), 1e-9),
        edgecolor="#272727", linewidth=0.6, zorder=3,
    )
    for _, r in top5.iterrows():
        ax_band.annotate(
            f"{int(r['banda_nm'])}", (DIAS.index(r["dia"]), r["banda_nm"]),
            xytext=(5, (int(r["posicao_dia"]) - 3) * 9), textcoords="offset points", fontsize=8,
        )
    ax_band.set(xticks=x, xticklabels=DIAS, ylabel="Comprimento de onda (nm)")
    ax_band.set_ylim(
        max(350, top5["banda_nm"].min() - 100),
        top5["banda_nm"].max() + 260,
    )
    ax_band.set_title("Evolução das 5 bandas significativas mais importantes no RF", loc="left", weight="bold")
    ax_band.grid(axis="y", alpha=0.35)
    barra = fig.colorbar(pontos, ax=ax_band, pad=0.01)
    barra.set_label("Importância por permutação")
    fig.savefig(saida / f"evolucao_rf_{genotipo}_manha.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
