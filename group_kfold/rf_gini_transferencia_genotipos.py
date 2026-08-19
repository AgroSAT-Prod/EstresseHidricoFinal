#!/usr/bin/env python3
"""Transferência de RF entre genótipos com seleção Spearman + Gini.

Em cada dia, cada genótipo de origem seleciona suas cinco bandas de maior
importância Gini após reduzir colinearidade por Spearman |r| > 0,80 em janela
de 10 nm. A união dessas bandas treina a RF nos dois materiais de origem; o
terceiro genótipo é um teste externo completo.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "testeDeNormalidade"), str(ROOT / "reducaoColinearidade")])
from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

SAIDA = ROOT / "group_kfold" / "outputs" / "transferencia_rf_gini_r080"
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
CENARIOS = {
    "BR16+CD202→EMB48": (["BR16", "CD202"], "EMB48"),
    "EMB48+CD202→BR16": (["EMB48", "CD202"], "BR16"),
    "BR16+EMB48→CD202": (["BR16", "EMB48"], "CD202"),
}
SEEDS, TOP_K = (42, 43, 44, 45, 46), 5


def selecionar_gini(X, y, w):
    corr = spearman_matriz(X)
    grupos = agrupar(corr, w, limiar_r=.80, janela_nm=10.0)
    reps = np.array([idx[len(idx) // 2] for g in np.unique(grupos) if (idx := np.flatnonzero(grupos == g)).size])
    modelo = RandomForestClassifier(n_estimators=100, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=42).fit(X[:, reps], y)
    tabela = pd.DataFrame({"banda_nm": w[reps].astype(int), "importancia_gini": modelo.feature_importances_}).sort_values("importancia_gini", ascending=False).reset_index(drop=True)
    tabela.insert(0, "posicao", np.arange(1, len(tabela) + 1))
    return tabela.head(TOP_K), len(reps)


def metricas(y, pred, score):
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred), "kappa": cohen_kappa_score(y, pred), "auc_roc": roc_auc_score(y, score)}


def resumo(df, grupos):
    return (df.groupby(grupos, as_index=False)[["accuracy", "f1", "kappa", "auc_roc"]]
              .agg(["mean", "std", "var"]).reset_index().pipe(lambda x: x.set_axis(["_".join(c).strip("_") for c in x.columns], axis=1)))


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    meta, espectro, w = carregar("normalizado", turno="manha")
    selecionadas, resultados, predicoes = [], [], []
    for dia in DIAS:
        mask_dia = meta.dia.eq(dia).to_numpy()
        meta_dia, X_dia = meta.loc[mask_dia].reset_index(drop=True), espectro[mask_dia]
        for nome, (origens, alvo) in CENARIOS.items():
            bandas_origem = []
            for origem in origens:
                mask = meta_dia.genotipo.eq(origem).to_numpy()
                y_origem = meta_dia.loc[mask, "condicao"].eq("NIRRIG").astype(int).to_numpy()
                top, n_reps = selecionar_gini(X_dia[mask], y_origem, w)
                top.insert(0, "dia", dia); top.insert(0, "cenario", nome); top.insert(0, "genotipo_origem", origem); top["representantes_spearman"] = n_reps
                selecionadas.append(top); bandas_origem.extend(top.banda_nm.tolist())
            bandas = sorted(set(map(int, bandas_origem)))
            mapa = {int(b): i for i, b in enumerate(w.astype(int))}
            idx = np.array([mapa[b] for b in bandas])
            train = meta_dia.genotipo.isin(origens).to_numpy(); teste = meta_dia.genotipo.eq(alvo).to_numpy()
            Xtr, Xte = X_dia[train][:, idx], X_dia[teste][:, idx]
            ytr = meta_dia.loc[train, "condicao"].eq("NIRRIG").astype(int).to_numpy()
            yte = meta_dia.loc[teste, "condicao"].eq("NIRRIG").astype(int).to_numpy()
            for seed in SEEDS:
                modelo = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=seed).fit(Xtr, ytr)
                pred, score = modelo.predict(Xte), modelo.predict_proba(Xte)[:, 1]
                linha = {"cenario": nome, "dia": dia, "genotipos_origem": "+".join(origens), "genotipo_teste": alvo, "seed": seed, "n_treino": len(ytr), "n_teste": len(yte), "n_bandas": len(bandas), "bandas_nm": ";".join(map(str, bandas))}
                linha.update(metricas(yte, pred, score)); resultados.append(linha)
                predicoes.append(pd.DataFrame({"cenario": nome, "dia": dia, "seed": seed, "genotipo_teste": alvo, "classe_real_nirrig": yte, "classe_predita_nirrig": pred, "score_nirrig": score}))
            print(f"{dia} | {nome.replace('→', '->')}: {len(bandas)} bandas", flush=True)
    df_sel, df_res = pd.concat(selecionadas, ignore_index=True), pd.DataFrame(resultados)
    df_sel.to_csv(SAIDA / "top5_gini_por_origem.csv", sep=";", index=False); df_res.to_csv(SAIDA / "metricas_por_seed.csv", sep=";", index=False); pd.concat(predicoes, ignore_index=True).to_csv(SAIDA / "predicoes.csv", sep=";", index=False)
    por_dia = resumo(df_res, ["cenario", "dia", "genotipos_origem", "genotipo_teste"])
    geral = resumo(df_res, ["cenario", "genotipos_origem", "genotipo_teste"])
    por_dia.to_csv(SAIDA / "metricas_por_dia.csv", sep=";", index=False); geral.to_csv(SAIDA / "metricas_resumo_cenarios.csv", sep=";", index=False)
    linhas = ["# Transferência entre genótipos — Random Forest", "", "Seleção feita apenas nos dois genótipos de origem: Spearman |r| > 0,80 (janela 10 nm), seguida das Top 5 bandas por importância Gini de cada origem. A RF é treinada nas duas origens e testada exclusivamente no terceiro genótipo. Valores são média ± DP de cinco sementes, agregados nos sete dias.", "", "| Treino → teste | Acc | F1 | κ | AUC-ROC |", "|---|---:|---:|---:|---:|"]
    for r in geral.itertuples():
        linhas.append(f"| {r.cenario} | {r.accuracy_mean:.3f} ± {r.accuracy_std:.3f} | {r.f1_mean:.3f} ± {r.f1_std:.3f} | {r.kappa_mean:.3f} ± {r.kappa_std:.3f} | {r.auc_roc_mean:.3f} ± {r.auc_roc_std:.3f} |")
    linhas.extend(["", "## Resultados dia a dia", ""])
    for cenario, (origens, alvo) in CENARIOS.items():
        linhas.extend([f"### {cenario}", "", "| Dia | Top 5 da origem 1 | Top 5 da origem 2 | Bandas usadas (união) | Acc | F1 | κ | AUC-ROC |", "|---|---|---|---|---:|---:|---:|---:|"])
        for dia in DIAS:
            listas = []
            for origem in origens:
                valores = df_sel[(df_sel.cenario == cenario) & (df_sel.dia == dia) & (df_sel.genotipo_origem == origem)].sort_values("posicao")
                listas.append(", ".join(str(int(v)) for v in valores.banda_nm))
            r = por_dia[(por_dia.cenario == cenario) & (por_dia.dia == dia)].iloc[0]
            n_bandas = df_res[(df_res.cenario == cenario) & (df_res.dia == dia)].n_bandas.iloc[0]
            linhas.append(f"| {dia} | {listas[0]} | {listas[1]} | {n_bandas:.0f} | {r.accuracy_mean:.3f} ± {r.accuracy_std:.3f} | {r.f1_mean:.3f} ± {r.f1_std:.3f} | {r.kappa_mean:.3f} ± {r.kappa_std:.3f} | {r.auc_roc_mean:.3f} ± {r.auc_roc_std:.3f} |")
        linhas.append("")
    (SAIDA / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    pd.DataFrame([{ "spearman": "|r| > 0,80", "janela_nm": 10, "ranking": "importância Gini", "top_k_por_origem": TOP_K, "n_arvores": 500, "seeds": ", ".join(map(str, SEEDS)), "validacao": "teste externo por genótipo; média de 5 sementes", "turno": "manha", "estagio": "normalizado" }]).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)


if __name__ == "__main__": main()
