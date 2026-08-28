#!/usr/bin/env python3
"""PLS-DA/VIP e RF/Gini por dia, reunindo os três genótipos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_student

from plsda_vip_kfold4_tres_genotipos import analisar as analisar_pls, SEEDS as SEEDS_PLS, TOP_K
from rf_spearman_kfold4_tres_genotipos import analisar as analisar_rf, SEEDS as SEEDS_RF
from shapiro_normalidade import carregar

ROOT = Path(__file__).resolve().parent.parent.parent
SAIDA = ROOT / "analisePorDia" / "resultados" / "analiseAgrupada" / "dataset_gerado_spearman_r080"
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
COR = "#7A5195"


def resumir(folds, dia):
    linha = {"genotipo": "todos", "dia": dia, "n_dobras": len(folds)}
    for metrica in ["accuracy", "f1", "kappa", "auc_roc"]:
        valores = folds[metrica]
        media, dp = valores.mean(), valores.std(ddof=1)
        margem = t_student.ppf(.975, df=len(valores) - 1) * dp / np.sqrt(len(valores))
        minimo = -1.0 if metrica == "kappa" else 0.0
        linha.update({metrica: media, f"{metrica}_dp": dp, f"{metrica}_variancia": valores.var(ddof=1),
                      f"{metrica}_ic95_inf": max(minimo, media - margem), f"{metrica}_ic95_sup": min(1.0, media + margem)})
    return linha


def grafico(top5, metodo, ranking):
    fig, ax = plt.subplots(figsize=(13, 7))
    y_pos = {dia: y for y, dia in enumerate(reversed(DIAS))}
    for dia in DIAS:
        valores = sorted(top5.loc[top5.dia == dia, "banda_nm"].astype(int))
        y = y_pos[dia]
        ax.plot([min(valores), max(valores)], [y, y], color=COR, linewidth=10, alpha=.20, solid_capstyle="round")
        ax.scatter(valores, [y] * len(valores), color=COR, s=75, edgecolor="white", linewidth=1.2, zorder=3)
        for i, valor in enumerate(valores):
            dy = 10 if i % 2 == 0 else -16
            ax.annotate(str(valor), (valor, y), xytext=(0, dy), textcoords="offset points", ha="center",
                        va="bottom" if dy > 0 else "top", fontsize=8, color=COR, fontweight="bold")
    ax.set_yticks(list(y_pos.values()), list(y_pos.keys())); ax.set_ylabel("Dia de avaliação")
    ax.set_xlim(300, 2050); ax.set_ylim(-.6, len(DIAS)-.4); ax.set_xlabel("Comprimento de onda (nm)")
    ax.grid(axis="x", linestyle="--", alpha=.30); ax.grid(axis="y", linestyle=":", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Top 5 bandas para separar Irrig × NIrrig — {metodo} ({ranking})", y=.98, fontsize=15, fontweight="bold")
    fig.text(.5, .945, "Genótipos agrupados; Spearman |r| > 0,80, janela 10 nm; StratifiedKFold (k = 4), 5 sementes", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .90)); return fig


def relatorio(top5, metricas, metodo, ranking):
    linhas = [f"# {metodo} — Irrig × NIrrig com genótipos agrupados", "", "BR16, CD202 e EMB48 foram reunidos em cada dia. Spearman |r| > 0,80 em janela de 10 nm; StratifiedKFold (k=4), sem GroupKFold, repetido nas sementes 42–46 (20 dobras/dia).", "", f"Top 5 por {ranking}.", "", "| Dia | Top 5 (nm; ranking) | Acc (IC95%) | F1 (IC95%) | κ (IC95%) | AUC-ROC (IC95%) |", "|---|---|---:|---:|---:|---:|"]
    for dia in DIAS:
        t = top5[top5.dia == dia]
        coluna = "vip" if ranking == "VIP" else "importancia_gini"
        bandas = ", ".join(f"{int(x.banda_nm)} ({getattr(x, coluna):.3f})" for x in t.itertuples())
        m = metricas[metricas.dia == dia].iloc[0]
        f = lambda n: f"{m[n]:.3f} [{m[n + '_ic95_inf']:.3f}; {m[n + '_ic95_sup']:.3f}]"
        linhas.append(f"| {dia} | {bandas} | {f('accuracy')} | {f('f1')} | {f('kappa')} | {f('auc_roc')} |")
    return "\n".join(linhas) + "\n"


def executar(nome, analisar, arquivo_importancia, metodo, ranking):
    meta, X, w = carregar("normalizado", turno="manha")
    tops, metricas, configs = [], [], []
    for dia in DIAS:
        print(nome, dia, flush=True)
        imp, folds, pred, *resto = analisar(meta, X, w, "todos", dia, criterio="r")
        amostras, reps = resto[-2:]
        pasta = SAIDA / nome / dia; pasta.mkdir(parents=True, exist_ok=True)
        imp.to_csv(pasta / arquivo_importancia, sep=";", index=False); folds.to_csv(pasta / "metricas_folds.csv", sep=";", index=False); pred.to_csv(pasta / "predicoes.csv", sep=";", index=False)
        top = imp.head(TOP_K).copy(); top.insert(0, "dia", dia); top.insert(0, "genotipo", "todos"); tops.append(top)
        metricas.append(resumir(folds, dia)); configs.append({"dia": dia, "genotipos": "BR16, CD202, EMB48", "amostras": amostras, "representantes_spearman": reps, "criterio_spearman": "|r| > 0,80", "limiar_r": .80, "janela_nm": 10, "validacao": "StratifiedKFold", "k": 4, "seeds": ", ".join(map(str, SEEDS_PLS)), "n_dobras": len(folds), "ranking": ranking})
    top5, met = pd.concat(tops, ignore_index=True), pd.DataFrame(metricas)
    destino = SAIDA / nome
    top5.to_csv(destino / "top5_bandas_por_dia.csv", sep=";", index=False); met.to_csv(destino / "metricas_por_dia.csv", sep=";", index=False); pd.DataFrame(configs).to_csv(destino / "configuracao.csv", sep=";", index=False)
    (destino / "relatorio.md").write_text(relatorio(top5, met, metodo, ranking), encoding="utf-8")
    fig = grafico(top5, metodo, ranking); fig.savefig(destino / "top5_bandas_por_dia.png", dpi=300, bbox_inches="tight"); fig.savefig(destino / "top5_bandas_por_dia.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    executar("plsda_vip", analisar_pls, "vip_bandas.csv", "PLS-DA", "VIP")
    executar("random_forest_gini", analisar_rf, "importancia_bandas.csv", "Random Forest", "importância Gini")


if __name__ == "__main__": main()
