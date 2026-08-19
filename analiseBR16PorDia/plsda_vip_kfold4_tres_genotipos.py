#!/usr/bin/env python3
"""PLS-DA com seleção de Top 5 VIP após Spearman, por genótipo e dia."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_student
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "testeDeNormalidade"), str(ROOT / "reducaoColinearidade")])
from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, agrupar_por_pvalor, spearman_matriz  # noqa: E402

SAIDA = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "plsda_vip_spearman_p0001_5seeds_kfold4"
GENOTIPOS, DIAS = ["BR16", "CD202", "EMB48"], ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
CORES = {"BR16": "#4C78A8", "CD202": "#F58518", "EMB48": "#54A24B"}
SEEDS, N_COMPONENTES, TOP_K = (42, 43, 44, 45, 46), 2, 5


def ajustar_pls(X, y):
    return PLSRegression(n_components=min(N_COMPONENTES, X.shape[1], X.shape[0] - 1), scale=True).fit(X, y.astype(float))


def vip(pls):
    """Variable Importance in Projection para PLS com resposta binária."""
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total = s.sum()
    return np.sqrt(W.shape[0] * ((W ** 2) @ s) / total) if total else np.zeros(W.shape[0])


def analisar(meta, X, w, genotipo, dia, criterio="p"):
    mascara = meta.dia.eq(dia) if genotipo == "todos" else (meta.genotipo.eq(genotipo) & meta.dia.eq(dia))
    Xd = X[mascara.to_numpy()]
    y = meta.loc[mascara, "condicao"].eq("NIRRIG").astype(int).to_numpy()
    corr = spearman_matriz(Xd)
    grupos = (agrupar(corr, w, limiar_r=0.80, janela_nm=10.0) if criterio == "r"
              else agrupar_por_pvalor(corr, w, n_amostras=len(y), p_limiar=0.0001, janela_nm=10.0))
    reps = np.array([idx[len(idx) // 2] for g in np.unique(grupos) if (idx := np.flatnonzero(grupos == g)).size])
    Xr, bandas = Xd[:, reps], w[reps].astype(int)

    dobras, predicoes, vips_folds = [], [], []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(Xr, y), 1):
            # A seleção VIP ocorre somente no treino de cada dobra.
            selecao = ajustar_pls(Xr[tr], y[tr])
            vip_treino = vip(selecao)
            idx_top = np.argsort(vip_treino)[::-1][:TOP_K]
            modelo = ajustar_pls(Xr[tr][:, idx_top], y[tr])
            score = modelo.predict(Xr[te][:, idx_top]).ravel()
            pred = (score >= 0.5).astype(int)
            dobras.append({"seed": seed, "fold": fold, "n_treino": len(tr), "n_teste": len(te), "accuracy": accuracy_score(y[te], pred),
                           "f1": f1_score(y[te], pred), "kappa": cohen_kappa_score(y[te], pred), "auc_roc": roc_auc_score(y[te], score)})
            predicoes.append(pd.DataFrame({"seed": seed, "fold": fold, "classe_real_nirrig": y[te], "classe_predita_nirrig": pred, "score_plsda": score}))
            vips_folds.append(pd.DataFrame({"seed": seed, "fold": fold, "banda_nm": bandas, "vip": vip_treino}))

    # VIP final, exclusivamente descritivo, para a lista e a figura de Top 5.
    final = ajustar_pls(Xr, y)
    imp = pd.DataFrame({"banda_nm": bandas, "vip": vip(final)}).sort_values("vip", ascending=False).reset_index(drop=True)
    imp.insert(0, "posicao", np.arange(1, len(imp) + 1))
    return imp, pd.DataFrame(dobras), pd.concat(predicoes), pd.concat(vips_folds), len(y), len(bandas)


def plotar(top5):
    fig, eixos = plt.subplots(1, 3, figsize=(18, 7), sharex=True, sharey=True)
    ypos = {dia: i for i, dia in enumerate(reversed(DIAS))}
    for ax, g in zip(eixos, GENOTIPOS):
        for d in DIAS:
            valores = sorted(top5.loc[(top5.genotipo == g) & (top5.dia == d), "banda_nm"].astype(int))
            y, cor = ypos[d], CORES[g]
            ax.plot([min(valores), max(valores)], [y, y], color=cor, linewidth=10, alpha=.20, solid_capstyle="round")
            ax.scatter(valores, [y] * len(valores), color=cor, s=68, edgecolor="white", linewidth=1.1, zorder=3)
            for i, valor in enumerate(valores):
                dy = 9 if i % 2 == 0 else -15
                ax.annotate(str(valor), (valor, y), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom" if dy > 0 else "top", fontsize=7.5, color=cor, fontweight="bold")
        ax.set_title(g, fontweight="bold", color=CORES[g]); ax.set_yticks(list(ypos.values()), list(ypos.keys()))
        ax.set_xlim(300, 2050); ax.set_ylim(-.6, len(DIAS)-.4); ax.set_xlabel("Comprimento de onda (nm)")
        ax.grid(axis="x", linestyle="--", alpha=.3); ax.grid(axis="y", linestyle=":", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    eixos[0].set_ylabel("Dia de avaliação")
    fig.suptitle("Top 5 bandas para separar Irrig × NIrrig por PLS-DA (VIP)", y=.99, fontsize=15, fontweight="bold")
    fig.text(.5, .945, "Uma linha por dia; Spearman p < 0,0001, janela 10 nm; StratifiedKFold (k = 4)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .90)); return fig


def relatorio(top5, metricas):
    linhas = ["# PLS-DA — Irrig × NIrrig", "", "Spearman com p < 0,0001 (0,01%) em janela de 10 nm. A validação usa StratifiedKFold (k=4), sem GroupKFold, repetido em cinco sementes (42–46): 20 dobras por genótipo×dia. Em cada dobra, as cinco bandas VIP são selecionadas apenas no treino antes da classificação.", ""]
    for g in GENOTIPOS:
        linhas += [f"## {g}", "", "| Dia | Top 5 (nm; VIP final) | Acc (IC95%) | F1 (IC95%) | κ (IC95%) | AUC-ROC (IC95%) |", "|---|---|---:|---:|---:|---:|"]
        for d in DIAS:
            t = top5[(top5.genotipo == g) & (top5.dia == d)]
            texto = ", ".join(f"{int(x.banda_nm)} ({x.vip:.3f})" for x in t.itertuples())
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
            print(g, d, flush=True)
            imp, folds, pred, vip_folds, n, reps = analisar(meta, X, w, g, d)
            pasta = SAIDA / g / d; pasta.mkdir(parents=True, exist_ok=True)
            imp.to_csv(pasta / "vip_bandas.csv", sep=";", index=False); folds.to_csv(pasta / "metricas_folds.csv", sep=";", index=False)
            pred.to_csv(pasta / "predicoes.csv", sep=";", index=False); vip_folds.to_csv(pasta / "vip_por_fold.csv", sep=";", index=False)
            t = imp.head(TOP_K).copy(); t.insert(0, "dia", d); t.insert(0, "genotipo", g); tops.append(t)
            linha = {"genotipo": g, "dia": d, "n_dobras": len(folds)}
            for c in ["accuracy", "f1", "kappa", "auc_roc"]:
                valores = folds[c]
                media, dp = valores.mean(), valores.std(ddof=1)
                erro = dp / np.sqrt(len(valores))
                margem = t_student.ppf(.975, df=len(valores) - 1) * erro
                limite_inf = -1.0 if c == "kappa" else 0.0
                limite_sup = 1.0
                linha.update({c: media, f"{c}_dp": dp, f"{c}_variancia": valores.var(ddof=1),
                              f"{c}_ic95_inf": max(limite_inf, media - margem), f"{c}_ic95_sup": min(limite_sup, media + margem)})
            metricas.append(linha)
            configs.append({"genotipo":g, "dia":d, "amostras":n, "bandas_originais":len(w), "representantes_spearman":reps, "criterio_spearman":"p < 0,0001", "p_limiar":.0001, "janela_nm":10, "validacao":"StratifiedKFold", "k":4, "seeds":", ".join(map(str, SEEDS)), "n_dobras":len(folds), "n_componentes":N_COMPONENTES, "ranking":"VIP"})
    top5, met = pd.concat(tops, ignore_index=True), pd.DataFrame(metricas)
    top5.to_csv(SAIDA / "top5_bandas_vip_por_dia_genotipo.csv", sep=";", index=False); met.to_csv(SAIDA / "metricas_plsda_por_dia_genotipo.csv", sep=";", index=False); pd.DataFrame(configs).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)
    (SAIDA / "relatorio_top5_vip.md").write_text(relatorio(top5, met), encoding="utf-8")
    fig = plotar(top5); fig.savefig(SAIDA / "top5_bandas_vip_por_dia_genotipo.png", dpi=300, bbox_inches="tight"); fig.savefig(SAIDA / "top5_bandas_vip_por_dia_genotipo.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__": main()
