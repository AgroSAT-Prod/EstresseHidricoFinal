#!/usr/bin/env python3
"""Top 5 bandas do efeito de estresse por dia e por genótipo.

Para cada combinação de dia e genótipo, usa o contraste já calculado entre
IRRIG e NIRRIG em ``comparacao_estresse.csv``. Mantém somente bandas com
``p_valor <= 0,05`` e ordena as cinco menores p-values.

Uso:
    python top5_estresse_por_dia_genotipo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT.parent / "resultados" / "dataset_gerado" / "comparacao_estresse.csv"
SAIDA = ROOT.parent / "resultados" / "dataset_gerado" / "top5_bandas_estresse_por_dia_genotipo.csv"

ALPHA = 0.05
TOP_K = 5


def selecionar_top5(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona cinco bandas por dia e genótipo pelo menor p-valor."""
    partes = []
    for (dia, genotipo), sub in df.groupby(["dia", "genotipo"], sort=True):
        top = sub[sub["p_valor"] <= ALPHA].sort_values(
            ["p_valor", "banda_nm"], kind="stable"
        ).head(TOP_K).copy()
        if top.empty:
            continue
        top.insert(0, "posicao", np.arange(1, len(top) + 1))
        top.insert(0, "condicao_b", "NIRRIG")
        top.insert(0, "condicao_a", "IRRIG")
        partes.append(top[[
            "dia", "genotipo", "condicao_a", "condicao_b", "posicao",
            "banda_nm", "p_valor", "q_fdr", "H", "epsilon2",
            "delta_cliff", "n_irrig", "n_nirrig",
        ]])

    colunas = [
        "dia", "genotipo", "condicao_a", "condicao_b", "posicao",
        "banda_nm", "p_valor", "q_fdr", "H", "epsilon2",
        "delta_cliff", "n_irrig", "n_nirrig",
    ]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=colunas)


def main() -> None:
    if not ENTRADA.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {ENTRADA}")

    dados = pd.read_csv(ENTRADA, sep=";")
    top5 = selecionar_top5(dados)
    top5.to_csv(SAIDA, sep=";", index=False)

    n_grupos = top5[["dia", "genotipo"]].drop_duplicates().shape[0]
    print(f"{len(top5)} bandas em {n_grupos} combinações dia × genótipo.")
    print(f"Resultado salvo em {SAIDA}")


if __name__ == "__main__":
    main()
