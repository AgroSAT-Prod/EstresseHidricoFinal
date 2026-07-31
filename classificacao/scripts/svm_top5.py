#!/usr/bin/env python3
"""Support Vector Machine com bandas gerais ou por genotipo."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .classificacao_utils import SEMENTE, criar_parser, executar_modelo
except ImportError:
    from classificacao_utils import SEMENTE, criar_parser, executar_modelo


def criar_modelo() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=SEMENTE,
        )),
    ])


def extras_svm(modelo: Pipeline, dados: dict[str, object]) -> dict[str, pd.DataFrame]:
    importancia = permutation_importance(
        modelo,
        dados["X_teste"],
        dados["y_teste"],
        n_repeats=30,
        random_state=SEMENTE + int(dados["fold"]),
        scoring="accuracy",
    )

    svm = modelo.named_steps["modelo"]
    classes = list(svm.classes_)
    suportes = {f"n_support_{classe}": svm.n_support_[i] for i, classe in enumerate(classes)}

    return {
        "importancia_permutacao.csv": pd.DataFrame({
            "banda_nm": dados["bandas"],
            "importancia_media": importancia.importances_mean,
            "importancia_desvio": importancia.importances_std,
        }).sort_values("importancia_media", ascending=False),
        "parametros_modelo.csv": pd.DataFrame([{
            "kernel": svm.kernel,
            "C": svm.C,
            "gamma": svm.gamma,
            **suportes,
        }]),
    }


def main() -> None:
    parser = criar_parser("Support Vector Machine com bandas selecionadas.")
    args = parser.parse_args()
    executar_modelo(
        "svm",
        criar_modelo(),
        modo=args.modo,
        estagio=args.estagio,
        salvar_csv=args.salvar_csv,
        extras_fn=extras_svm,
    )


if __name__ == "__main__":
    main()
