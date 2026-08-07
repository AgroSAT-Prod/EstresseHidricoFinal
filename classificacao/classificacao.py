#!/usr/bin/env python3
"""Executa todos os modelos de classificacao com bandas selecionadas."""

from __future__ import annotations

import pandas as pd

from scripts.classificacao_utils import criar_parser, executar_modelo
from scripts.gradient_boosting_top5 import (
    criar_modelo as criar_gradient_boosting,
    extras_gradient_boosting,
)
from scripts.pls_da_top5 import criar_modelo as criar_pls_da, extras_pls_da
from scripts.random_forest_top5 import criar_modelo as criar_random_forest, extras_random_forest
from scripts.svm_top5 import criar_modelo as criar_svm, extras_svm


MODELOS = [
    ("random_forest", criar_random_forest, extras_random_forest),
    ("gradient_boosting", criar_gradient_boosting, extras_gradient_boosting),
    ("svm", criar_svm, extras_svm),
    ("pls_da", criar_pls_da, extras_pls_da),
]


def main() -> None:
    parser = criar_parser("Executa Random Forest, Gradient Boosting, SVM e PLS-DA.")
    args = parser.parse_args()
    resumos = []

    for nome, criar_modelo, extras_fn in MODELOS:
        print("\n" + "=" * 72)
        resumos.append(
            executar_modelo(
                nome,
                criar_modelo(),
                modo=args.modo,
                estagio=args.estagio,
                salvar_csv=args.salvar_csv,
                extras_fn=extras_fn,
            )
        )

    print("\nResumo final:")
    df_resumo = pd.concat(resumos, ignore_index=True)
    print(df_resumo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
