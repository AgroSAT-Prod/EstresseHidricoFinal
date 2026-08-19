#!/usr/bin/env python3
"""Spearman (p < 0,0001; janela 10 nm) + RF com K-fold estratificado.

Executa separadamente para cada genótipo e dia, classificando IRRIG versus
NIRRIG. A redução Spearman é não supervisionada e feita antes da RF em cada
combinação genótipo×dia. A validação é StratifiedKFold com 4 dobras, sem usar
o identificador de bloco como grupo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_student
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "testeDeNormalidade"), str(ROOT / "reducaoColinearidade")])
from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, agrupar_por_pvalor, spearman_matriz  # noqa: E402

SAIDA = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "rf_gini_spearman_p0001_5seeds_kfold4"
GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
CORES = {"BR16": "#4C78A8", "CD202": "#F58518", "EMB48": "#54A24B"}
SEEDS, N_ARVORES = (42, 43, 44, 45, 46), 100


def analisar(meta, X, w, genotipo, dia, criterio="p"):
    recorte = meta["dia"].eq(dia) if genotipo == "todos" else (meta["genotipo"].eq(genotipo) & meta["dia"].eq(dia))
    meta_dia, X_dia = meta.loc[recorte].reset_index(drop=True), X[recorte.to_numpy()]
    y = meta_dia["condicao"].eq("NIRRIG").astype(int).to_numpy()

    corr = spearman_matriz(X_dia)
    grupos = (agrupar(corr, w, limiar_r=0.80, janela_nm=10.0) if criterio == "r"
              else agrupar_por_pvalor(corr, w, n_amostras=len(y), p_limiar=0.0001, janela_nm=10.0))
    reps = np.array([
        idx[len(idx) // 2] for grupo in np.unique(grupos)
        if (idx := np.flatnonzero(grupos == grupo)).size
    ])
    Xr, bandas = X_dia[:, reps], w[reps].astype(int)

    dobras, predicoes = [], []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        for fold, (treino, teste) in enumerate(cv.split(Xr, y), start=1):
            modelo = RandomForestClassifier(
                n_estimators=N_ARVORES, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=1, n_jobs=1, random_state=seed * 100 + fold,
            ).fit(Xr[treino], y[treino])
            pred, prob = modelo.predict(Xr[teste]), modelo.predict_proba(Xr[teste])[:, 1]
            dobras.append({"seed": seed, "fold": fold, "n_treino": len(treino), "n_teste": len(teste),
                           "accuracy": accuracy_score(y[teste], pred), "f1": f1_score(y[teste], pred),
                           "kappa": cohen_kappa_score(y[teste], pred), "auc_roc": roc_auc_score(y[teste], prob)})
            predicoes.append(pd.DataFrame({"seed": seed, "fold": fold, "condicao_real": meta_dia.loc[teste, "condicao"],
                                           "classe_real_nirrig": y[teste], "classe_predita_nirrig": pred,
                                           "probabilidade_nirrig": prob}))

    final = RandomForestClassifier(
        n_estimators=N_ARVORES, max_features="sqrt", class_weight="balanced",
        min_samples_leaf=1, n_jobs=1, random_state=SEEDS[0],
    ).fit(Xr, y)
    imp = pd.DataFrame({
        "banda_nm": bandas,
        "importancia_gini": final.feature_importances_,
    }).sort_values("importancia_gini", ascending=False).reset_index(drop=True)
    imp.insert(0, "posicao", np.arange(1, len(imp) + 1))
    return imp, pd.DataFrame(dobras), pd.concat(predicoes, ignore_index=True), len(meta_dia), len(bandas)


def plotar(top5):
    fig, eixos = plt.subplots(1, 3, figsize=(18, 7), sharex=True, sharey=True)
    posicoes_y = {dia: y for y, dia in enumerate(reversed(DIAS))}
    for eixo, genotipo in zip(eixos, GENOTIPOS):
        dados = top5[top5["genotipo"] == genotipo]
        for dia in DIAS:
            valores = sorted(dados.loc[dados["dia"] == dia, "banda_nm"].astype(int))
            y, cor = posicoes_y[dia], CORES[genotipo]
            eixo.plot([min(valores), max(valores)], [y, y], color=cor, linewidth=10,
                      alpha=0.20, solid_capstyle="round")
            eixo.scatter(valores, [y] * len(valores), color=cor, s=68, edgecolor="white",
                         linewidth=1.1, zorder=3)
            for i, valor in enumerate(valores):
                dy = 9 if i % 2 == 0 else -15
                eixo.annotate(str(valor), (valor, y), xytext=(0, dy), textcoords="offset points",
                              ha="center", va="bottom" if dy > 0 else "top", fontsize=7.5,
                              color=cor, fontweight="bold")
        eixo.set_title(genotipo, fontweight="bold", color=CORES[genotipo])
        eixo.set_yticks(list(posicoes_y.values()), list(posicoes_y.keys()))
        eixo.set_xlim(300, 2050); eixo.set_ylim(-0.6, len(DIAS) - 0.4)
        eixo.grid(axis="x", linestyle="--", alpha=0.30); eixo.grid(axis="y", linestyle=":", alpha=0.25)
        eixo.spines[["top", "right"]].set_visible(False); eixo.set_xlabel("Comprimento de onda (nm)")
    eixos[0].set_ylabel("Dia de avaliação")
    fig.suptitle("Top 5 bandas para separar Irrig × NIrrig por Random Forest", y=0.99, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, "Uma linha por dia; Spearman p < 0,0001, janela 10 nm; StratifiedKFold (k = 4)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def relatorio(top5, metricas):
    linhas = ["# Random Forest — Irrig × NIrrig", "", "Spearman com p < 0,0001 (0,01%) em janela de 10 nm e StratifiedKFold com k = 4 (sem GroupKFold), repetido em cinco sementes (42–46): 20 dobras por genótipo×dia. As Top 5 são ordenadas pela importância Gini.", ""]
    for g in GENOTIPOS:
        linhas.extend([f"## {g}", "", "| Dia | Top 5 (nm; importância Gini) | Acc (IC95%) | F1 (IC95%) | κ (IC95%) | AUC-ROC (IC95%) |", "|---|---|---:|---:|---:|---:|"])
        for d in DIAS:
            bandas = top5[(top5.genotipo == g) & (top5.dia == d)]
            texto = ", ".join(f"{int(x.banda_nm)} ({x.importancia_gini:.3f})" for x in bandas.itertuples())
            m = metricas[(metricas.genotipo == g) & (metricas.dia == d)].iloc[0]
            fmt = lambda nome: f"{m[nome]:.3f} [{m[nome + '_ic95_inf']:.3f}; {m[nome + '_ic95_sup']:.3f}]"
            linhas.append(f"| {d} | {texto} | {fmt('accuracy')} | {fmt('f1')} | {fmt('kappa')} | {fmt('auc_roc')} |")
        linhas.append("")
    return "\n".join(linhas) + "\n"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    meta, X, w = carregar("normalizado", turno="manha")
    tops, metricas, configs = [], [], []
    for g in GENOTIPOS:
        for d in DIAS:
            print(f"{g} {d}", flush=True)
            pasta = SAIDA / g / d
            pasta.mkdir(parents=True, exist_ok=True)
            arquivos = [pasta / nome for nome in ("importancia_bandas.csv", "metricas_folds.csv", "predicoes.csv")]
            if all(arquivo.exists() for arquivo in arquivos):
                print("  resultado existente reaproveitado", flush=True)
                imp = pd.read_csv(arquivos[0], sep=";").sort_values("importancia_gini", ascending=False).reset_index(drop=True)
                imp["posicao"] = np.arange(1, len(imp) + 1)
                imp.to_csv(arquivos[0], sep=";", index=False)
                dobras = pd.read_csv(arquivos[1], sep=";")
                pred = pd.read_csv(arquivos[2], sep=";")
                n, reps = len(pred), len(imp)
            else:
                imp, dobras, pred, n, reps = analisar(meta, X, w, g, d)
                imp.to_csv(pasta / "importancia_bandas.csv", sep=";", index=False)
                dobras.to_csv(pasta / "metricas_folds.csv", sep=";", index=False)
                pred.to_csv(pasta / "predicoes.csv", sep=";", index=False)
            top = imp.head(5).copy(); top.insert(0, "dia", d); top.insert(0, "genotipo", g); tops.append(top)
            resumo = {"genotipo": g, "dia": d, "n_dobras": len(dobras)}
            for c in ["accuracy", "f1", "kappa", "auc_roc"]:
                valores = dobras[c]
                media, dp = valores.mean(), valores.std(ddof=1)
                margem = t_student.ppf(.975, df=len(valores) - 1) * dp / np.sqrt(len(valores))
                limite_inf = -1.0 if c == "kappa" else 0.0
                resumo.update({c: media, f"{c}_dp": dp, f"{c}_variancia": valores.var(ddof=1),
                               f"{c}_ic95_inf": max(limite_inf, media - margem),
                               f"{c}_ic95_sup": min(1.0, media + margem)})
            metricas.append(resumo)
            configs.append({"genotipo": g, "dia": d, "amostras": n, "bandas_originais": len(w), "representantes_spearman": reps,
                            "criterio_spearman": "p < 0,0001", "p_limiar": 0.0001, "janela_nm": 10, "validacao": "StratifiedKFold", "k": 4,
                            "seeds": ", ".join(map(str, SEEDS)), "n_dobras": len(dobras), "n_arvores": N_ARVORES,
                            "criterio_ranking_top5": "importancia_gini"})
    top5, met = pd.concat(tops, ignore_index=True), pd.DataFrame(metricas)
    top5.to_csv(SAIDA / "top5_bandas_rf_por_dia_genotipo.csv", sep=";", index=False)
    met.to_csv(SAIDA / "metricas_rf_por_dia_genotipo.csv", sep=";", index=False)
    pd.DataFrame(configs).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)
    (SAIDA / "relatorio_top5_rf.md").write_text(relatorio(top5, met), encoding="utf-8")
    fig = plotar(top5)
    fig.savefig(SAIDA / "top5_bandas_rf_por_dia_genotipo.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "top5_bandas_rf_por_dia_genotipo.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
