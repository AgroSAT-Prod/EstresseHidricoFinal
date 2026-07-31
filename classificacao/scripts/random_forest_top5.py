#!/usr/bin/env python3
"""Random Forest com bandas gerais ou por genotipo."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from .classificacao_utils import SEMENTE, criar_parser, executar_modelo
except ImportError:
    from classificacao_utils import SEMENTE, criar_parser, executar_modelo


def criar_modelo() -> Pipeline:
    return Pipeline([
        ("modelo", RandomForestClassifier(
            n_estimators=500,
            random_state=SEMENTE,
            class_weight="balanced",
            n_jobs=1,
        )),
    ])


def extras_random_forest(modelo: Pipeline, dados: dict[str, object]) -> dict[str, pd.DataFrame]:
    rf = modelo.named_steps["modelo"]
    return {
        "importancia_bandas.csv": pd.DataFrame({
            "banda_nm": dados["bandas"],
            "importancia": rf.feature_importances_,
        }).sort_values("importancia", ascending=False),
    }


def main() -> None:
    parser = criar_parser("Random Forest com bandas selecionadas.")
    args = parser.parse_args()
    executar_modelo(
        "random_forest",
        criar_modelo(),
        modo=args.modo,
        estagio=args.estagio,
        salvar_csv=args.salvar_csv,
        extras_fn=extras_random_forest,
    )


if __name__ == "__main__":
    main()
