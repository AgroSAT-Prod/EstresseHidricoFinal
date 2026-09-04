#!/usr/bin/env python3
"""Random Forest para BR16/D02, manhã: IRRIG versus NIRRIG.

As 2051 bandas são primeiro reduzidas por Spearman (|rho| > 0,80, grupos
contíguos de no máximo 10 nm). O RF usa uma representante por grupo; a
avaliação é leave-one-block-out para impedir que leituras da mesma parcela
entrem simultaneamente em treino e teste. A importância por permutação é
calculada em cada bloco deixado de fora e depois promediada.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))
from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

SAIDA_BASE = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "genotipo" / "BR16"
COMPARACAO = ROOT / "testeDiferencaSignificativa" / "resultados" / "dataset_gerado" / "comparacao_estresse.csv"
SEMENTE = 42
N_ARVORES = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="RF BR16 por dia, turno da manhã.")
    parser.add_argument("--genotipo", default="BR16", choices=("BR16", "CD202", "EMB48"))
    parser.add_argument("--dia", default="D02", choices=("D02", "D03", "D04", "D05", "D06", "D09", "D10"))
    args = parser.parse_args()
    dia = args.dia
    genotipo = args.genotipo
    saida = SAIDA_BASE.parent / genotipo / dia / "random_forest"
    meta, X, w = carregar("normalizado", turno="manha")
    mask = (meta["genotipo"].eq(genotipo) & meta["dia"].eq(dia)).to_numpy()
    meta, X = meta.loc[mask].reset_index(drop=True), X[mask]
    y = meta["condicao"].eq("NIRRIG").astype(int).to_numpy()
    grupos = meta["bloco"].astype(str).to_numpy()

    # A representante é a banda central de cada grupo, sem usar o alvo na
    # redução. Assim, a etapa de Spearman é estritamente não supervisionada.
    corr = spearman_matriz(X)
    ids = agrupar(corr, w)
    reps = np.array([
        idx[len(idx) // 2] for g in np.unique(ids)
        if (idx := np.flatnonzero(ids == g)).size
    ])
    Xr, bandas = X[:, reps], w[reps].astype(int)

    cv = GroupKFold(n_splits=len(np.unique(grupos)))
    linhas, predicoes, importancias = [], [], []
    for fold, (treino, teste) in enumerate(cv.split(Xr, y, groups=grupos), start=1):
        modelo = RandomForestClassifier(
            n_estimators=N_ARVORES, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=1, n_jobs=1, random_state=SEMENTE + fold,
        ).fit(Xr[treino], y[treino])
        pred = modelo.predict(Xr[teste])
        prob = modelo.predict_proba(Xr[teste])[:, 1]
        linhas.append({
            "fold": fold, "bloco_teste": np.unique(grupos[teste])[0], "n_teste": len(teste),
            "accuracy": accuracy_score(y[teste], pred),
            "balanced_accuracy": balanced_accuracy_score(y[teste], pred),
            "precision": precision_score(y[teste], pred, zero_division=0),
            "recall": recall_score(y[teste], pred, zero_division=0),
            "f1": f1_score(y[teste], pred, zero_division=0),
            "auc_roc": roc_auc_score(y[teste], prob),
        })
        predicoes.append(pd.DataFrame({
            "fold": fold, "bloco": grupos[teste], "condicao_real": meta.loc[teste, "condicao"],
            "classe_real_nirrig": y[teste], "classe_predita_nirrig": pred,
            "probabilidade_nirrig": prob,
        }))
        perm = permutation_importance(
            modelo, Xr[teste], y[teste], scoring="balanced_accuracy",
            n_repeats=3, random_state=SEMENTE + fold, n_jobs=1,
        )
        importancias.append(perm.importances_mean)

    # Ajuste final apenas para a importância Gini descritiva; os scores acima
    # são exclusivamente os de validação por bloco.
    final = RandomForestClassifier(
        n_estimators=N_ARVORES, max_features="sqrt", class_weight="balanced",
        min_samples_leaf=1, n_jobs=1, random_state=SEMENTE,
    ).fit(Xr, y)
    estat = pd.read_csv(COMPARACAO, sep=";")
    estat = estat[(estat["genotipo"] == genotipo) & (estat["dia"] == dia)].set_index("banda_nm")
    imp = pd.DataFrame({
        "banda_nm": bandas,
        "importancia_permutacao_media": np.mean(importancias, axis=0),
        "importancia_permutacao_dp": np.std(importancias, axis=0),
        "importancia_gini": final.feature_importances_,
    }).join(estat[["p_valor", "q_fdr", "epsilon2", "delta_cliff", "significativa"]], on="banda_nm")
    imp["abs_delta_cliff"] = imp["delta_cliff"].abs()
    imp = imp.sort_values(["importancia_permutacao_media", "importancia_gini"], ascending=False)
    imp.insert(0, "posicao", np.arange(1, len(imp) + 1))
    top = imp.head(10)

    metricas = pd.DataFrame(linhas)
    resumo = metricas.drop(columns=["fold", "bloco_teste", "n_teste"]).agg(["mean", "std"]).T.reset_index()
    resumo.columns = ["metrica", "media", "desvio"]
    saida.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(saida / "metricas_folds.csv", sep=";", index=False)
    resumo.to_csv(saida / "metricas_resumo.csv", sep=";", index=False)
    pd.concat(predicoes, ignore_index=True).to_csv(saida / "predicoes.csv", sep=";", index=False)
    imp.to_csv(saida / "importancia_bandas.csv", sep=";", index=False)
    top.to_csv(saida / "top10_bandas.csv", sep=";", index=False)
    pd.DataFrame([{
        "genotipo": genotipo, "dia": dia, "turno": "manha", "amostras": len(meta),
        "bandas_originais": len(w), "representantes_spearman": len(bandas),
        "limiar_spearman_abs_rho": 0.80, "n_arvores": N_ARVORES,
        "validacao": "GroupKFold leave-one-block-out", "classe_positiva": "NIRRIG",
    }]).to_csv(saida / "configuracao.csv", sep=";", index=False)

    print("Scores de validação (média das 4 dobras):")
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTop 10 bandas por importância por permutação:")
    print(top[["posicao", "banda_nm", "importancia_permutacao_media", "importancia_gini", "q_fdr", "delta_cliff"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
