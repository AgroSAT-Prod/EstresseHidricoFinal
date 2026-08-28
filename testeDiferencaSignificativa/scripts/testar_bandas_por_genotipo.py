#!/usr/bin/env python3
"""Testa bandas relevantes para cada genótipo sob estresse hídrico.

Para cada genótipo (BR16, CD202, EMB48), filtra as amostras e aplica o
pipeline de seleção de bandas usando condição (IRRIG vs NIRRIG) como label.
Identifica quais regiões espectrais são mais discriminantes para cada
genótipo sob estresse hídrico.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "preprocessamento_espectral"))

from preprocessamento import (  # noqa: E402
    ENTRADA,
    LIMITE_INF,
    LIMITE_SUP,
    interpolar,
    jump_correct,
    limpar_valores_espurios,
    SPLICE_POINTS,
    REFERENCE_SEQ,
)

from selecao_estatistica import (  # noqa: E402
    fdr_bh,
    kruskal_per_band,
    make_windows,
    pick_best_per_window,
    select_top_k_lowcollinear,
    USAR_JANELAS,
    WINDOW_BANDS,
    K_FINAL,
    CORR_THRESH,
    MIN_SEP_NM,
    Q_MAX,
)

SAIDA_DIR = ROOT.parent.parent / "dataset"

GENOTIPOS = ["BR16", "CD202", "EMB48"]


def preprocessar(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Aplica pré-processamento completo e retorna espectro + comprimentos de onda."""
    bandas = sorted(int(c) for c in df.columns if c.isdigit())
    bandas_str = [str(b) for b in bandas]

    mask = [(b >= LIMITE_INF and b <= LIMITE_SUP) for b in bandas]
    bandas_validas = [b for b, m in zip(bandas, mask) if m]
    w = np.array(bandas_validas, dtype=float)

    espectro = df[bandas_str].values.astype(float)
    espectro_valido = espectro[:, mask]

    espectro_limpo = limpar_valores_espurios(espectro_valido)

    espectro_corrigido = np.zeros_like(espectro_limpo)
    for i in range(espectro_limpo.shape[0]):
        s = pd.Series(espectro_limpo[i], index=w, dtype="float64")
        s = jump_correct(s, SPLICE_POINTS, REFERENCE_SEQ)
        espectro_corrigido[i] = s.to_numpy()

    espectro_interp = interpolar(espectro_corrigido)

    return espectro_interp, w


def selecionar_por_genotipo(
    df: pd.DataFrame,
    genotipo: str,
    espectro: np.ndarray,
    w: np.ndarray,
) -> pd.DataFrame:
    """Seleciona bandas relevantes para um genótipo específico."""
    mask_gen = df["genotipo"] == genotipo
    df_gen = df[mask_gen]
    espectro_gen = espectro[mask_gen.values]

    labels = df_gen["condicao"].astype(str).to_numpy()

    pvals = kruskal_per_band(espectro_gen, labels)
    qvals = np.clip(fdr_bh(pvals), 0, 1)

    valid_mask = (
        np.isfinite(qvals) & np.isfinite(pvals)
        & np.isfinite(espectro_gen).all(axis=0)
    )

    if valid_mask.sum() < 2:
        print(f"  {genotipo}: bandas insuficientes")
        return pd.DataFrame()

    Ysub = espectro_gen[:, valid_mask]
    w_sub = w[valid_mask]
    pvals_sub = pvals[valid_mask]
    qvals_sub = qvals[valid_mask]

    candidate_idx = None
    if USAR_JANELAS:
        candidate_idx = pick_best_per_window(qvals_sub, make_windows(len(w_sub), WINDOW_BANDS))

    sel = select_top_k_lowcollinear(
        Ysub, w_sub, qvals_sub, candidate_idx=candidate_idx,
        k=K_FINAL, corr_thresh=CORR_THRESH, min_sep_nm=MIN_SEP_NM, q_max=Q_MAX,
    )

    resultados = []
    for rank, j in enumerate(sel, 1):
        resultados.append({
            "genotipo": genotipo,
            "rank": rank,
            "banda_nm": int(w_sub[j]),
            "p_valor": float(pvals_sub[j]),
            "q_fdr": float(qvals_sub[j]),
        })

    return pd.DataFrame(resultados)


def main() -> None:
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    print(f"Carregando {len(df)} amostras...")
    espectro, w = preprocessar(df)
    print(f"Pré-processamento concluído: {len(w)} bandas ({LIMITE_INF}-{LIMITE_SUP} nm)")

    todos_resultados = []

    for genotipo in GENOTIPOS:
        n_amostras = (df["genotipo"] == genotipo).sum()
        print(f"\n{genotipo} ({n_amostras} amostras):")

        df_resultados = selecionar_por_genotipo(df, genotipo, espectro, w)

        if not df_resultados.empty:
            print(f"  Selecionadas {len(df_resultados)} bandas:")
            for _, row in df_resultados.iterrows():
                print(f"    {row['banda_nm']:4d} nm  "
                      f"(p={row['p_valor']:.2e}, q={row['q_fdr']:.2e})")

            df_resultados.to_csv(
                SAIDA_DIR / f"bandas_{genotipo}.csv",
                sep=";",
                index=False,
            )
            todos_resultados.append(df_resultados)

    if todos_resultados:
        df_consolidado = pd.concat(todos_resultados, ignore_index=True)
        df_consolidado.to_csv(
            SAIDA_DIR / "bandas_por_genotipo.csv",
            sep=";",
            index=False,
        )
        print(f"\nResultados consolidados salvos em {SAIDA_DIR / 'bandas_por_genotipo.csv'}")


if __name__ == "__main__":
    main()
