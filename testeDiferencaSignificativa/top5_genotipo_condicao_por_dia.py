#!/usr/bin/env python3
"""Top 5 bandas por dia, genótipo e condição.

Em cada dia e condição, cada genótipo é comparado separadamente aos demais
genótipos. Para uma banda, retém-se o contraste com menor p-valor; as cinco
bandas com ``p_valor <= 0,05`` são então ordenadas pelo menor p-valor.

Uso:
    python top5_genotipo_condicao_por_dia.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / "dataset_gerado" / "efeitos_simples.csv"
SAIDA = ROOT / "dataset_gerado" / "top5_bandas_por_dia_genotipo_condicao.csv"

ALPHA = 0.05
TOP_K = 5


def contrastes_por_genotipo(df: pd.DataFrame) -> pd.DataFrame:
    """Reexpressa cada contraste para os dois genótipos que ele compara."""
    comum = [
        "dia", "condicao", "banda_nm", "teste", "H", "p_kruskal",
        "q_kruskal", "U", "p_mannwhitney", "q_mannwhitney", "p_valor",
        "q_fdr", "epsilon2",
    ]
    a = df[comum + ["genotipo_a", "genotipo_b", "n_a", "n_b", "delta_cliff"]].copy()
    a = a.rename(columns={
        "genotipo_a": "genotipo", "genotipo_b": "genotipo_referencia",
        "n_a": "n_genotipo", "n_b": "n_referencia",
    })

    b = df[comum + ["genotipo_a", "genotipo_b", "n_a", "n_b", "delta_cliff"]].copy()
    b = b.rename(columns={
        "genotipo_b": "genotipo", "genotipo_a": "genotipo_referencia",
        "n_b": "n_genotipo", "n_a": "n_referencia",
    })
    # Delta de Cliff deve manter a direção genótipo focal − referência.
    b["delta_cliff"] = -b["delta_cliff"]
    return pd.concat([a, b], ignore_index=True)


def selecionar_top5(df: pd.DataFrame) -> pd.DataFrame:
    """Top 5 de cada dia × genótipo × condição pelo menor p-valor."""
    chaves_banda = ["dia", "genotipo", "condicao", "banda_nm"]
    melhor_contraste = (
        df.sort_values(["p_valor", "genotipo_referencia"], kind="stable")
        .groupby(chaves_banda, as_index=False)
        .first()
    )

    partes = []
    for (dia, genotipo, condicao), sub in melhor_contraste.groupby(
        ["dia", "genotipo", "condicao"], sort=True
    ):
        top = sub[sub["p_valor"] <= ALPHA].sort_values(
            ["p_valor", "banda_nm"], kind="stable"
        ).head(TOP_K).copy()
        if top.empty:
            continue
        top.insert(3, "posicao", np.arange(1, len(top) + 1))
        partes.append(top)

    colunas = ["dia", "genotipo", "condicao", "posicao", "banda_nm"]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=colunas)


def main() -> None:
    if not ENTRADA.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {ENTRADA}")

    efeitos = pd.read_csv(ENTRADA, sep=";")
    top5 = selecionar_top5(contrastes_por_genotipo(efeitos))
    top5.to_csv(SAIDA, sep=";", index=False)

    n_grupos = top5[["dia", "genotipo", "condicao"]].drop_duplicates().shape[0]
    print(f"{len(top5)} bandas em {n_grupos} combinações dia × genótipo × condição.")
    print(f"Resultado salvo em {SAIDA}")


if __name__ == "__main__":
    main()
