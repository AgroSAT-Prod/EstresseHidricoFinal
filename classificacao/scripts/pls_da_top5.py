#!/usr/bin/env python3
"""PLS-DA com bandas gerais ou por genotipo."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from .classificacao_utils import PLSDAClassifier, criar_parser, executar_modelo
except ImportError:
    from classificacao_utils import PLSDAClassifier, criar_parser, executar_modelo


def criar_modelo() -> Pipeline:
    return Pipeline([
        ("modelo", PLSDAClassifier(n_components=2)),
    ])


def extras_pls_da(modelo: Pipeline, dados: dict[str, object]) -> dict[str, pd.DataFrame]:
    pls_da = modelo.named_steps["modelo"]
    pls = pls_da.modelo_

    pesos = pd.DataFrame(
        pls.x_weights_,
        columns=[f"componente_{i + 1}" for i in range(pls_da.n_components_)],
    )
    pesos.insert(0, "banda_nm", dados["bandas"])

    coeficientes = pd.DataFrame({
        "banda_nm": dados["bandas"],
        "coeficiente": pls.coef_.ravel(),
    }).sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)

    return {
        "pesos_pls.csv": pesos,
        "coeficientes_pls.csv": coeficientes,
        "parametros_modelo.csv": pd.DataFrame([{
            "n_componentes": pls_da.n_components_,
        }]),
    }


def main() -> None:
    parser = criar_parser("PLS-DA com bandas selecionadas.")
    args = parser.parse_args()
    executar_modelo(
        "pls_da",
        criar_modelo(),
        modo=args.modo,
        estagio=args.estagio,
        salvar_csv=args.salvar_csv,
        extras_fn=extras_pls_da,
    )


if __name__ == "__main__":
    main()
