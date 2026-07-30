#!/usr/bin/env python3
"""Seleção estatística de bandas via Kruskal-Wallis + FDR.

Lê o espectro pré-processado e normalizado, aplica:
1. Kruskal-Wallis banda a banda (teste não-paramétrico entre grupos)
2. Correção FDR (Benjamini-Hochberg)
3. Seleção de 1 candidato por janela espectral
4. Seleção greedy das K bandas mais significativas com baixa colinearidade
   (|Spearman| < CORR_THRESH, separadas por pelo menos MIN_SEP_NM)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_normalizado.csv"
SAIDA = ROOT.parent / "dataset" / "Unificada13052026_selecionadas.csv"

USAR_JANELAS = True
WINDOW_BANDS = 10
K_FINAL = 10
Q_MAX = 0.05
CORR_THRESH = 0.85
MIN_SEP_NM = 10.0
MIN_N_PER_GROUP = 2


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (FDR) sem libs externas."""
    p = np.array(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if len(idx) == 0:
        return q
    pv = p[idx]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, len(pv_sorted) + 1)
    q_sorted = pv_sorted * len(pv_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    back = np.empty_like(pv)
    back[order] = q_sorted
    q[idx] = back
    return q


def make_windows(n_bands: int, window_bands: int) -> list[tuple[int, int]]:
    return [(i, min(i + window_bands, n_bands)) for i in range(0, n_bands, window_bands)]


def pick_best_per_window(qvals: np.ndarray, windows: list[tuple[int, int]]) -> list[int]:
    reps = []
    for a, b in windows:
        seg = qvals[a:b]
        if np.all(~np.isfinite(seg)):
            continue
        reps.append(a + np.nanargmin(seg))
    return reps


def abs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """|Spearman| via ranks + Pearson nos ranks."""
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - np.nanmean(ra)
    rb = rb - np.nanmean(rb)
    denom = np.sqrt(np.nansum(ra**2)) * np.sqrt(np.nansum(rb**2))
    if denom == 0 or not np.isfinite(denom):
        return 1.0
    return abs(np.nansum(ra * rb) / denom)


def kruskal_per_band(Ysub: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """p-valor do Kruskal-Wallis banda a banda."""
    labels = np.asarray(labels)
    idxs = [np.where(labels == u)[0] for u in pd.unique(labels)]

    pvals = np.full(Ysub.shape[1], np.nan, dtype=float)
    for j in range(Ysub.shape[1]):
        samples = []
        for idx in idxs:
            v = Ysub[idx, j]
            v = v[np.isfinite(v)]
            if len(v) >= MIN_N_PER_GROUP:
                samples.append(v)
        if len(samples) < 2:
            continue
        try:
            p = kruskal(*samples).pvalue
        except Exception:
            continue
        if np.isfinite(p) and 0.0 <= p <= 1.0:
            pvals[j] = p

    return pvals


def select_top_k_lowcollinear(
    Ysub: np.ndarray,
    w: np.ndarray,
    qvals: np.ndarray,
    candidate_idx: np.ndarray | None = None,
    *,
    k: int,
    corr_thresh: float,
    min_sep_nm: float,
    q_max: float,
) -> list[int]:
    """Seleciona K bandas com baixa colinearidade e separação mínima."""
    q = np.array(qvals, dtype=float)
    if candidate_idx is None:
        candidate_idx = np.arange(len(w))
    candidate_idx = np.array(candidate_idx, dtype=int)

    ord_idx = candidate_idx[np.argsort(q[candidate_idx])]
    selected: list[int] = []

    for j in ord_idx:
        if len(selected) >= k:
            break
        if not np.isfinite(q[j]) or q[j] > q_max:
            continue

        wl = w[j]
        if any(abs(wl - w[s]) < min_sep_nm for s in selected):
            continue

        if all(abs_spearman(Ysub[:, j], Ysub[:, s]) < corr_thresh for s in selected):
            selected.append(j)

    return selected


def selecao_estatistica(
    df: pd.DataFrame,
    labels_col: str = "condicao",
) -> tuple[pd.DataFrame, list[int]]:
    """Aplica seleção estatística de bandas."""
    meta = [c for c in df.columns if not c.isdigit()]
    bandas = sorted(int(c) for c in df.columns if c.isdigit())
    bandas_str = [str(b) for b in bandas]
    w = np.array(bandas, dtype=float)

    print(f"Seleção estatística em {len(bandas)} bandas...")

    espectro = df[bandas_str].values.astype(float)

    labels = df[labels_col].astype(str).to_numpy()
    pvals = kruskal_per_band(espectro, labels)
    qvals = np.clip(fdr_bh(pvals), 0, 1)

    valid_mask = (
        np.isfinite(qvals) & np.isfinite(pvals)
        & np.isfinite(espectro).all(axis=0)
    )
    if valid_mask.sum() < 2:
        print("Bandas insuficientes para seleção estatística.")
        return df, []

    Ysub = espectro[:, valid_mask]
    w_sub = w[valid_mask]
    qvals_sub = qvals[valid_mask]

    candidate_idx = None
    if USAR_JANELAS:
        candidate_idx = pick_best_per_window(qvals_sub, make_windows(len(w_sub), WINDOW_BANDS))

    sel = select_top_k_lowcollinear(
        Ysub, w_sub, qvals_sub, candidate_idx=candidate_idx,
        k=K_FINAL, corr_thresh=CORR_THRESH, min_sep_nm=MIN_SEP_NM, q_max=Q_MAX,
    )

    bandas_selecionadas = [int(w_sub[j]) for j in sel]
    print(f"Selecionadas {len(bandas_selecionadas)} bandas: {bandas_selecionadas}")

    bandas_selecionadas_str = [str(b) for b in bandas_selecionadas]
    df_saida = df[meta].copy()
    df_saida[bandas_selecionadas_str] = espectro[:, valid_mask][:, sel]

    return df_saida, bandas_selecionadas


def main() -> None:
    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    df_selecionado, bandas = selecao_estatistica(df, labels_col="condicao")

    df_selecionado.to_csv(SAIDA, sep=";", index=False)
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
