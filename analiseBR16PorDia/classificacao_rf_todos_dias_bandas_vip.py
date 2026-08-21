#!/usr/bin/env python3
"""Random Forest por genótipo com as Top 5 VIP dos sete dias reunidos."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

ENTRADA = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "plsda_vip_5seeds_kfold4_todos_dias"
SAIDA = ENTRADA / "classificacao_random_forest"
GENOTIPOS = ("BR16", "CD202", "EMB48")
SEMENTE, K_FOLDS = 42, 5
CLASSE_POSITIVA = "NIRRIG"


def selecionar_colunas(espectro: np.ndarray, w: np.ndarray, bandas: list[int]) -> np.ndarray:
    posicao = {int(nm): i for i, nm in enumerate(w.astype(int))}
    ausentes = [b for b in bandas if b not in posicao]
    if ausentes:
        raise ValueError(f"Bandas ausentes do espectro: {ausentes}")
    return espectro[:, [posicao[b] for b in bandas]].astype(float)


def modelo_base() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, random_state=SEMENTE, class_weight="balanced", n_jobs=1)


def metricas(y_real: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_bin = (y_real == CLASSE_POSITIVA).astype(int)
    return {"accuracy": accuracy_score(y_real, y_pred),
            "precision": precision_score(y_real, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0),
            "recall": recall_score(y_real, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0),
            "f1_score": f1_score(y_real, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0),
            "kappa": cohen_kappa_score(y_real, y_pred), "auc_roc": roc_auc_score(y_bin, score)}


def analisar(meta: pd.DataFrame, X: np.ndarray, bandas: list[int], genotipo: str):
    y = meta.condicao.astype(str).to_numpy()
    grupos = meta.nomenclaura.astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    linhas, predicoes, importancias, atribuicoes = [], [], [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=grupos), start=1):
        modelo = clone(modelo_base()).fit(X[tr], y[tr])
        y_pred = modelo.predict(X[te])
        posicao_nirrig = list(modelo.classes_).index(CLASSE_POSITIVA)
        score = modelo.predict_proba(X[te])[:, posicao_nirrig]
        linhas.append({"genotipo": genotipo, "fold": fold, "n_treino": len(tr), "n_teste": len(te), **metricas(y[te], y_pred, score)})
        pred = meta.iloc[te].copy()
        pred.insert(0, "fold", fold); pred.insert(0, "genotipo_avaliado", genotipo)
        pred["classe_real"] = y[te]; pred["classe_predita"] = y_pred; pred["score_nirrig"] = score; pred["acerto"] = y_pred == y[te]
        predicoes.append(pred)
        importancias.append(pd.DataFrame({"genotipo": genotipo, "fold": fold, "banda_nm": bandas, "importancia_gini": modelo.feature_importances_}))
        for split, indice in (("treino", tr), ("teste", te)):
            df = meta.iloc[indice][["nomenclaura", "bloco", "condicao", "dia"]].copy()
            df.insert(0, "split", split); df.insert(0, "fold", fold); df.insert(0, "genotipo", genotipo)
            atribuicoes.append(df)
    return pd.DataFrame(linhas), pd.concat(predicoes, ignore_index=True), pd.concat(importancias, ignore_index=True), pd.concat(atribuicoes, ignore_index=True)


def resumo(folds: pd.DataFrame, genotipo: str) -> dict:
    linha = {"genotipo": genotipo, "n_dobras": len(folds), "amostras": int(folds.n_teste.sum())}
    for coluna in ("accuracy", "precision", "recall", "f1_score", "kappa", "auc_roc"):
        linha[f"{coluna}_media"] = folds[coluna].mean()
        linha[f"{coluna}_desvio"] = folds[coluna].std(ddof=1)
    return linha


def relatorio(resumos: pd.DataFrame, bandas_por_genotipo: dict[str, list[int]]) -> str:
    linhas = ["# Random Forest — bandas VIP consolidadas por genótipo", "",
              "Cada modelo usa somente as cinco bandas Top VIP do respectivo genótipo, obtidas com os sete dias reunidos. A classificação é IRRIG × NIRRIG usando todas as observações da manhã de D02, D03, D04, D05, D06, D09 e D10.",
              "", "Validação: StratifiedGroupKFold (k=5; semente 42), com `nomenclaura` como grupo. Assim, as oito leituras de um mesmo arquivo permanecem inteiramente no treino ou no teste.",
              "", "| Genótipo | Bandas (nm) | Acc ± DP | F1 ± DP | κ ± DP | AUC-ROC ± DP |", "|---|---|---:|---:|---:|---:|"]
    for g in GENOTIPOS:
        r = resumos[resumos.genotipo.eq(g)].iloc[0]
        fmt = lambda campo: f"{r[campo + '_media']:.3f} ± {r[campo + '_desvio']:.3f}"
        linhas.append(f"| {g} | {', '.join(map(str, bandas_por_genotipo[g]))} | {fmt('accuracy')} | {fmt('f1_score')} | {fmt('kappa')} | {fmt('auc_roc')} |")
    return "\n".join(linhas) + "\n"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    top5 = pd.read_csv(ENTRADA / "top5_bandas_vip_todos_dias_por_genotipo.csv", sep=";")
    meta, espectro, w = carregar("normalizado", turno="manha")
    resumos, configuracoes, bandas_por_genotipo = [], [], {}
    for genotipo in GENOTIPOS:
        bandas = top5.loc[top5.genotipo.eq(genotipo)].sort_values("posicao").banda_nm.astype(int).tolist()
        if len(bandas) != 5 or len(set(bandas)) != 5:
            raise ValueError(f"Top 5 inválido para {genotipo}: {bandas}")
        mascara = meta.genotipo.eq(genotipo).to_numpy()
        metag = meta.loc[mascara].reset_index(drop=True)
        Xg = selecionar_colunas(espectro[mascara], w, bandas)
        folds, predicoes, importancia, atribuicoes = analisar(metag, Xg, bandas, genotipo)
        pasta = SAIDA / genotipo; pasta.mkdir(exist_ok=True)
        folds.to_csv(pasta / "metricas_folds.csv", sep=";", index=False)
        predicoes.to_csv(pasta / "predicoes.csv", sep=";", index=False)
        importancia.to_csv(pasta / "importancia_gini_por_fold.csv", sep=";", index=False)
        (importancia.groupby("banda_nm", as_index=False).agg(importancia_gini_media=("importancia_gini", "mean"), importancia_gini_dp=("importancia_gini", "std")).sort_values("importancia_gini_media", ascending=False).to_csv(pasta / "importancia_gini_resumo.csv", sep=";", index=False))
        atribuicoes.to_csv(pasta / "fold_assignments.csv", sep=";", index=False)
        r = resumo(folds, genotipo); pd.DataFrame([r]).to_csv(pasta / "metricas_resumo.csv", sep=";", index=False); resumos.append(r)
        bandas_por_genotipo[genotipo] = bandas
        configuracoes.append({"genotipo": genotipo, "amostras": len(metag), "bandas_nm": ", ".join(map(str, bandas)), "origem_bandas": "Top 5 VIP consolidado dos sete dias", "modelo": "RandomForestClassifier", "n_estimators": 500, "class_weight": "balanced", "validacao": "StratifiedGroupKFold", "k": K_FOLDS, "grupo": "nomenclaura", "semente": SEMENTE, "turno": "manha"})
        print(f"{genotipo}: Acc={r['accuracy_media']:.3f}, AUC={r['auc_roc_media']:.3f}", flush=True)
    df_resumos = pd.DataFrame(resumos)
    df_resumos.to_csv(SAIDA / "metricas_resumo_por_genotipo.csv", sep=";", index=False)
    pd.DataFrame(configuracoes).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)
    (SAIDA / "relatorio_random_forest.md").write_text(relatorio(df_resumos, bandas_por_genotipo), encoding="utf-8")


if __name__ == "__main__":
    main()
