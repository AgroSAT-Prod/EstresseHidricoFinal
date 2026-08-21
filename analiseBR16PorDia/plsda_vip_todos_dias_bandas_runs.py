#!/usr/bin/env python3
"""PLS-DA/VIP por genótipo, reunindo os sete dias nas bandas das runs diárias.

Para cada genótipo, a entrada de variáveis é a união das Top 5 bandas VIP das
sete análises diárias em ``plsda_vip_5seeds_kfold4``.  O novo VIP, portanto,
compara somente bandas já selecionadas pelas runs e usa todas as observações
da manhã de D02, D03, D04, D05, D06, D09 e D10 daquele genótipo.
"""

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
sys.path.extend([str(ROOT / "testeDeNormalidade")])
from shapiro_normalidade import carregar  # noqa: E402

ENTRADA_RUNS = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "plsda_vip_5seeds_kfold4"
SAIDA = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "plsda_vip_5seeds_kfold4_todos_dias"
GENOTIPOS = ("BR16", "CD202", "EMB48")
DIAS = ("D02", "D03", "D04", "D05", "D06", "D09", "D10")
SEEDS, N_COMPONENTES, TOP_K = (42, 43, 44, 45, 46), 2, 5
CORES = {"BR16": "#4C78A8", "CD202": "#F58518", "EMB48": "#54A24B"}


def ajustar_pls(X: np.ndarray, y: np.ndarray) -> PLSRegression:
    return PLSRegression(n_components=min(N_COMPONENTES, X.shape[1], X.shape[0] - 1), scale=True).fit(X, y.astype(float))


def vip(modelo: PLSRegression) -> np.ndarray:
    """Variable Importance in Projection para PLS com resposta binária."""
    T, W, Q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    soma_componentes = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total = soma_componentes.sum()
    return np.sqrt(W.shape[0] * ((W ** 2) @ soma_componentes) / total) if total else np.zeros(W.shape[0])


def bandas_das_runs() -> pd.DataFrame:
    runs = pd.read_csv(ENTRADA_RUNS / "top5_bandas_vip_por_dia_genotipo.csv", sep=";")
    esperado = {(g, d) for g in GENOTIPOS for d in DIAS}
    encontrado = set(runs[["genotipo", "dia"]].itertuples(index=False, name=None))
    faltantes = esperado - encontrado
    if faltantes:
        raise ValueError(f"Runs diárias ausentes: {sorted(faltantes)}")
    return runs[runs.genotipo.isin(GENOTIPOS) & runs.dia.isin(DIAS)].copy()


def analisar_genotipo(meta: pd.DataFrame, X: np.ndarray, w: np.ndarray, genotipo: str, bandas: np.ndarray):
    mascara = meta.genotipo.eq(genotipo) & meta.dia.isin(DIAS)
    Xg = X[mascara.to_numpy()]
    metag = meta.loc[mascara].reset_index(drop=True)
    y = metag.condicao.eq("NIRRIG").astype(int).to_numpy()
    indice_por_nm = {int(nm): i for i, nm in enumerate(w.astype(int))}
    idx = np.array([indice_por_nm[int(nm)] for nm in bandas])
    Xs = Xg[:, idx]

    dobras, predicoes, vip_folds = [], [], []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(Xs, y), start=1):
            selecao = ajustar_pls(Xs[tr], y[tr])
            vip_treino = vip(selecao)
            idx_top = np.argsort(vip_treino)[::-1][:TOP_K]
            modelo = ajustar_pls(Xs[tr][:, idx_top], y[tr])
            score = modelo.predict(Xs[te][:, idx_top]).ravel()
            pred = (score >= .5).astype(int)
            dobras.append({"seed": seed, "fold": fold, "n_treino": len(tr), "n_teste": len(te),
                           "accuracy": accuracy_score(y[te], pred), "f1": f1_score(y[te], pred),
                           "kappa": cohen_kappa_score(y[te], pred), "auc_roc": roc_auc_score(y[te], score)})
            predicoes.append(pd.DataFrame({"seed": seed, "fold": fold, "dia": metag.dia.iloc[te].to_numpy(),
                                            "classe_real_nirrig": y[te], "classe_predita_nirrig": pred,
                                            "score_plsda": score}))
            vip_folds.append(pd.DataFrame({"seed": seed, "fold": fold, "banda_nm": bandas, "vip": vip_treino}))

    final = ajustar_pls(Xs, y)
    importancia = pd.DataFrame({"banda_nm": bandas, "vip": vip(final)}).sort_values("vip", ascending=False).reset_index(drop=True)
    importancia.insert(0, "posicao", np.arange(1, len(importancia) + 1))
    return importancia, pd.DataFrame(dobras), pd.concat(predicoes, ignore_index=True), pd.concat(vip_folds, ignore_index=True), len(Xg)


def resumir_metricas(folds: pd.DataFrame, genotipo: str) -> dict:
    linha = {"genotipo": genotipo, "n_dobras": len(folds)}
    for nome in ("accuracy", "f1", "kappa", "auc_roc"):
        valores = folds[nome]
        media, dp = valores.mean(), valores.std(ddof=1)
        margem = t_student.ppf(.975, df=len(valores) - 1) * dp / np.sqrt(len(valores))
        linha.update({nome: media, f"{nome}_dp": dp, f"{nome}_variancia": valores.var(ddof=1),
                      f"{nome}_ic95_inf": max(-1.0 if nome == "kappa" else 0.0, media - margem),
                      f"{nome}_ic95_sup": min(1.0, media + margem)})
    return linha


def relatorio(top5: pd.DataFrame, metricas: pd.DataFrame, n_bandas: dict[str, int]) -> str:
    linhas = ["# PLS-DA — VIP entre as bandas selecionadas nas runs diárias", "",
              "Para cada genótipo, as bandas candidatas são a união das Top 5 VIP das sete runs por dia (D02, D03, D04, D05, D06, D09 e D10). O modelo reúne todos os sete dias, mantendo os genótipos separados e classificando IRRIG × NIRRIG.",
              "", "A validação usa StratifiedKFold (k=4), sem GroupKFold, repetida nas sementes 42–46 (20 dobras por genótipo). Em cada dobra, as cinco bandas VIP são escolhidas somente no treino. O ranking exibido é o VIP descritivo do ajuste com todas as observações.",
              "", "| Genótipo | Bandas candidatas | Top 5 (nm; VIP final) | Acc (IC95%) | F1 (IC95%) | κ (IC95%) | AUC-ROC (IC95%) |", "|---|---:|---|---:|---:|---:|---:|"]
    for genotipo in GENOTIPOS:
        t = top5[top5.genotipo.eq(genotipo)]
        m = metricas[metricas.genotipo.eq(genotipo)].iloc[0]
        fmt = lambda nome: f"{m[nome]:.3f} [{m[nome + '_ic95_inf']:.3f}; {m[nome + '_ic95_sup']:.3f}]"
        bandas = ", ".join(f"{int(x.banda_nm)} ({x.vip:.3f})" for x in t.itertuples())
        linhas.append(f"| {genotipo} | {n_bandas[genotipo]} | {bandas} | {fmt('accuracy')} | {fmt('f1')} | {fmt('kappa')} | {fmt('auc_roc')} |")
    return "\n".join(linhas) + "\n"


def plotar(top5: pd.DataFrame):
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, genotipo in zip(eixos, GENOTIPOS):
        dados = top5[top5.genotipo.eq(genotipo)].sort_values("vip")
        ax.barh(dados.banda_nm.astype(str), dados.vip, color=CORES[genotipo])
        ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_title(genotipo, fontweight="bold", color=CORES[genotipo]); ax.set_xlabel("VIP final")
        ax.grid(axis="x", linestyle=":", alpha=.35); ax.spines[["top", "right"]].set_visible(False)
    eixos[0].set_ylabel("Comprimento de onda (nm)")
    fig.suptitle("Top 5 VIP com os sete dias reunidos, por genótipo", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92))
    return fig


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    meta, X, w = carregar("normalizado", turno="manha")
    runs = bandas_das_runs()
    top5s, metricas, configuracoes, n_bandas = [], [], [], {}
    for genotipo in GENOTIPOS:
        bandas = np.sort(runs.loc[runs.genotipo.eq(genotipo), "banda_nm"].unique().astype(int))
        importancia, folds, predicoes, vip_folds, n_amostras = analisar_genotipo(meta, X, w, genotipo, bandas)
        pasta = SAIDA / genotipo; pasta.mkdir(exist_ok=True)
        importancia.to_csv(pasta / "vip_bandas.csv", sep=";", index=False)
        folds.to_csv(pasta / "metricas_folds.csv", sep=";", index=False)
        predicoes.to_csv(pasta / "predicoes.csv", sep=";", index=False)
        vip_folds.to_csv(pasta / "vip_por_fold.csv", sep=";", index=False)
        origem = runs.loc[runs.genotipo.eq(genotipo), ["dia", "posicao", "banda_nm", "vip"]].sort_values(["dia", "posicao"])
        origem.to_csv(pasta / "bandas_origem_runs_diarias.csv", sep=";", index=False)
        top = importancia.head(TOP_K).copy(); top.insert(0, "genotipo", genotipo); top5s.append(top)
        metricas.append(resumir_metricas(folds, genotipo)); n_bandas[genotipo] = len(bandas)
        configuracoes.append({"genotipo": genotipo, "dias": ", ".join(DIAS), "amostras": n_amostras,
                              "bandas_candidatas": len(bandas), "origem_bandas": "união das Top 5 VIP das runs diárias",
                              "validacao": "StratifiedKFold", "k": 4, "seeds": ", ".join(map(str, SEEDS)),
                              "n_dobras": len(folds), "n_componentes": N_COMPONENTES, "ranking": "VIP"})
        print(f"{genotipo}: {n_amostras} amostras, {len(bandas)} bandas candidatas", flush=True)

    top5, met = pd.concat(top5s, ignore_index=True), pd.DataFrame(metricas)
    top5.to_csv(SAIDA / "top5_bandas_vip_todos_dias_por_genotipo.csv", sep=";", index=False)
    met.to_csv(SAIDA / "metricas_plsda_todos_dias_por_genotipo.csv", sep=";", index=False)
    pd.DataFrame(configuracoes).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)
    (SAIDA / "relatorio_top5_vip_todos_dias.md").write_text(relatorio(top5, met, n_bandas), encoding="utf-8")
    fig = plotar(top5); fig.savefig(SAIDA / "top5_bandas_vip_todos_dias_por_genotipo.png", dpi=300, bbox_inches="tight"); fig.savefig(SAIDA / "top5_bandas_vip_todos_dias_por_genotipo.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
