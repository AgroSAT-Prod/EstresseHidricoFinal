#!/usr/bin/env python3
"""Consolida as Top 5 bandas da Random Forest por dia e genótipo.

Usa os resultados de RF já calculados após a redução não supervisionada de
colinearidade por Spearman (|rho| >= 0,80). A importância usada no ranking é
a importância por permutação média, avaliada por bloco deixado de fora.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "analiseBR16PorDia" / "dataset_gerado"
SAIDA = BASE / "random_forest_tres_genotipos"
GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
CORES = {"BR16": "#4C78A8", "CD202": "#F58518", "EMB48": "#54A24B"}


def carregar_resultados():
    top5, metricas, configuracao = [], [], []
    for genotipo in GENOTIPOS:
        for dia in DIAS:
            pasta = BASE / genotipo / dia / "random_forest"
            imp = pd.read_csv(pasta / "importancia_bandas.csv", sep=";").head(5)
            imp.insert(0, "dia", dia)
            imp.insert(0, "genotipo", genotipo)
            top5.append(imp)

            dobras = pd.read_csv(pasta / "metricas_folds.csv", sep=";")
            predicoes = pd.read_csv(pasta / "predicoes.csv", sep=";")
            kappas = predicoes.groupby("fold").apply(
                lambda x: cohen_kappa_score(x["classe_real_nirrig"], x["classe_predita_nirrig"]),
                include_groups=False,
            )
            dobras["kappa"] = dobras["fold"].map(kappas)
            linha = dobras[["accuracy", "balanced_accuracy", "f1", "auc_roc", "kappa"]].mean().to_dict()
            linha.update({f"{col}_dp": dobras[col].std() for col in ["accuracy", "balanced_accuracy", "f1", "auc_roc", "kappa"]})
            linha.update({"genotipo": genotipo, "dia": dia})
            metricas.append(linha)

            config = pd.read_csv(pasta / "configuracao.csv", sep=";")
            configuracao.append(config)

    return pd.concat(top5, ignore_index=True), pd.DataFrame(metricas), pd.concat(configuracao, ignore_index=True)


def grafico(top5):
    fig, eixos = plt.subplots(1, 3, figsize=(18, 7), sharex=True, sharey=True)
    limites = (300, 2050)

    posicoes_y = {dia: y for y, dia in enumerate(reversed(DIAS))}
    for eixo, genotipo in zip(eixos, GENOTIPOS):
        dados_genotipo = top5[top5["genotipo"] == genotipo]
        for dia in DIAS:
            valores = sorted(dados_genotipo.loc[dados_genotipo["dia"] == dia, "banda_nm"].astype(int))
            y = posicoes_y[dia]
            cor = CORES[genotipo]
            eixo.plot([min(valores), max(valores)], [y, y], color=cor, linewidth=10,
                      alpha=0.20, solid_capstyle="round")
            eixo.scatter(valores, [y] * len(valores), color=cor, s=68, edgecolor="white",
                         linewidth=1.1, zorder=3)
            for i, valor in enumerate(valores):
                deslocamento = 9 if i % 2 == 0 else -15
                eixo.annotate(str(valor), (valor, y), xytext=(0, deslocamento),
                              textcoords="offset points", ha="center",
                              va="bottom" if deslocamento > 0 else "top", fontsize=7.5,
                              color=cor, fontweight="bold")
        eixo.set_title(genotipo, fontweight="bold", color=CORES[genotipo])
        eixo.set_yticks(list(posicoes_y.values()))
        eixo.set_yticklabels(list(posicoes_y.keys()))
        eixo.set_xlim(*limites)
        eixo.set_ylim(-0.6, len(DIAS) - 0.4)
        eixo.grid(axis="x", linestyle="--", alpha=0.30)
        eixo.grid(axis="y", linestyle=":", alpha=0.25)
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.set_xlabel("Comprimento de onda (nm)")

    eixos[0].set_ylabel("Dia de avaliação")
    fig.suptitle("Top 5 bandas para separar Irrig × NIrrig por Random Forest", y=0.99,
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, "Uma linha por dia; seleção prévia: Spearman |ρ| ≥ 0,80; importância por permutação (GroupKFold por bloco)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def relatorio_markdown(top5, metricas, configuracao):
    linhas = [
        "# Random Forest — Irrig × NIrrig por genótipo e dia",
        "",
        "As bandas foram reduzidas separadamente em cada combinação genótipo×dia por Spearman (|ρ| ≥ 0,80). "
        "A Random Forest foi avaliada com GroupKFold, deixando um bloco de campo por vez fora. "
        "O ranking abaixo usa a importância média por permutação na acurácia balanceada.",
        "",
        "Importância igual a 0,000 indica que a permutação daquela banda não reduziu a acurácia balanceada, em média, "
        "nos blocos de teste; nesses empates, a ordenação usa a importância Gini como desempate.",
        "",
    ]
    for genotipo in GENOTIPOS:
        linhas.extend([f"## {genotipo}", "", "| Dia | Top 5 bandas (nm; importância por permutação) | Acc | F1 | κ | AUC-ROC |", "|---|---|---:|---:|---:|---:|"])
        for dia in DIAS:
            parte = top5[(top5["genotipo"] == genotipo) & (top5["dia"] == dia)]
            bandas = ", ".join(f"{int(r.banda_nm)} ({r.importancia_permutacao_media:.3f})" for r in parte.itertuples())
            m = metricas[(metricas["genotipo"] == genotipo) & (metricas["dia"] == dia)].iloc[0]
            linhas.append(
                f"| {dia} | {bandas} | {m.accuracy:.3f} | {m.f1:.3f} | {m.kappa:.3f} | {m.auc_roc:.3f} |"
            )
        linhas.append("")
    representantes = configuracao["representantes_spearman"]
    linhas.extend([
        "## Configuração",
        "",
        f"Foram usadas 2.051 bandas originais e {representantes.min()}–{representantes.max()} representantes por combinação após Spearman. "
        "O modelo usou 100 árvores, `max_features=sqrt`, balanceamento de classes e semente 42.",
    ])
    return "\n".join(linhas) + "\n"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    top5, metricas, configuracao = carregar_resultados()
    top5.to_csv(SAIDA / "top5_bandas_rf_por_dia_genotipo.csv", sep=";", index=False)
    metricas.to_csv(SAIDA / "metricas_rf_por_dia_genotipo.csv", sep=";", index=False)
    configuracao.to_csv(SAIDA / "configuracao_spearman_rf.csv", sep=";", index=False)
    (SAIDA / "relatorio_top5_rf.md").write_text(
        relatorio_markdown(top5, metricas, configuracao), encoding="utf-8"
    )
    fig = grafico(top5)
    fig.savefig(SAIDA / "top5_bandas_rf_por_dia_genotipo.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "top5_bandas_rf_por_dia_genotipo.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
