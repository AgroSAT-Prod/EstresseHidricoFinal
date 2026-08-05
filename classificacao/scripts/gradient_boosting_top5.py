#!/usr/bin/env python3
"""Gradient boosting com bandas gerais ou por genotipo.

Usa o `HistGradientBoostingClassifier` do scikit-learn -- gradient boosting
sobre histogramas, o mesmo algoritmo que o XGBoost roda com
`tree_method="hist"`, sem adicionar dependencia ao projeto.

Com cinco bandas e ~450 amostras por genotipo, o risco nao e capacidade e sim
sobreajuste, entao as arvores sao rasas (`max_depth=3`) e o boosting para
sozinho: `early_stopping` com 15% do treino como validacao interna. Essa
validacao sai de dentro da dobra de treino, nunca da de teste, entao nao vaza
informacao para a metrica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from .classificacao_utils import (
        CLASSE_POSITIVA, SEMENTE, criar_parser, executar_modelo,
    )
except ImportError:
    from classificacao_utils import (
        CLASSE_POSITIVA, SEMENTE, criar_parser, executar_modelo,
    )


class BoostingBalanceado(HistGradientBoostingClassifier):
    """Boosting que reequilibra as classes por peso de amostra.

    O `HistGradientBoostingClassifier` nao tem `class_weight` como a Random
    Forest; o equivalente e passar `sample_weight` no fit. As celulas do
    experimento sao quase balanceadas, mas os folds nao saem exatos, e manter
    o mesmo tratamento dos outros modelos deixa as metricas comparaveis.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight)


def criar_modelo() -> Pipeline:
    return Pipeline([
        ("modelo", BoostingBalanceado(
            max_iter=500,
            learning_rate=0.06,
            max_depth=3,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=SEMENTE,
        )),
    ])


def extras_gradient_boosting(
    modelo: Pipeline,
    dados: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Importancia por permutacao, medida na propria dobra de teste.

    O boosting por histograma nao expoe `feature_importances_`. A importancia
    por permutacao e a alternativa direta e diz o que interessa: quanto a
    acuracia cai quando aquela banda e embaralhada.
    """
    gb = modelo.named_steps["modelo"]
    resultado = permutation_importance(
        gb, dados["X_teste"], dados["y_teste"],
        n_repeats=20, random_state=SEMENTE, scoring="accuracy",
    )

    return {
        "importancia_permutacao.csv": pd.DataFrame({
            "banda_nm": dados["bandas"],
            "importancia": resultado.importances_mean,
            "desvio": resultado.importances_std,
        }).sort_values("importancia", ascending=False),
        "iteracoes_boosting.csv": pd.DataFrame([{
            "iteracoes_usadas": int(gb.n_iter_),
            "iteracoes_maximas": gb.max_iter,
            "classe_positiva": CLASSE_POSITIVA,
        }]),
    }


def main() -> None:
    parser = criar_parser("Gradient boosting com bandas selecionadas.")
    args = parser.parse_args()
    executar_modelo(
        "gradient_boosting",
        criar_modelo(),
        modo=args.modo,
        estagio=args.estagio,
        salvar_csv=args.salvar_csv,
        extras_fn=extras_gradient_boosting,
    )


if __name__ == "__main__":
    main()
