#!/usr/bin/env python3
"""Pré-processamento espectral: jump correction, limpeza, interpolação e recorte.

Pipeline:
1. Jump correction (shift por detector) nas emendas ~1000 nm e ~1800 nm
2. Limpeza de valores espúrios (clip + remoção de inf/-inf)
3. Interpolação de NaNs com extrapolação nas bordas
4. Recorte para 400-2450 nm (descarte das bordas do detector)

ASD FieldSpec 3: 3 detectores (VNIR, SWIR1, SWIR2) com emendas em ~1000 nm
e ~1800 nm (valores consolidados na literatura).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_Limpa.csv"
SAIDA = ROOT.parent / "dataset" / "Unificada13052026_400_2450.csv"

LIMITE_INF = 400
LIMITE_SUP = 2450

SPLICE_POINTS = (1000, 1800)
REFERENCE_SEQ = 1

CLIP_ABS = 1e4


def jump_correct(
    series: pd.Series,
    splices: tuple[int, ...] = SPLICE_POINTS,
    reference_seq: int = REFERENCE_SEQ,
) -> pd.Series:
    """Corrige degraus nas emendas entre os 3 detectores do ASD via shift."""
    series = series.sort_index()

    def get_seq(wl: float) -> int:
        for i, s in enumerate(splices):
            if wl <= s:
                return i
        return len(splices)

    groups = series.groupby(get_seq)

    def shift(ref: pd.Series, mov: pd.Series, right: bool = True) -> pd.Series:
        diff = (ref.iloc[-1] - mov.iloc[0]) if right else (ref.iloc[0] - mov.iloc[-1])
        return mov + diff

    for i in range(reference_seq, groups.ngroups - 1):
        series.update(shift(groups.get_group(i), groups.get_group(i + 1), right=True))
    for i in range(reference_seq, 0, -1):
        series.update(shift(groups.get_group(i), groups.get_group(i - 1), right=False))

    return series


def limpar_valores_espurios(espectro: np.ndarray) -> np.ndarray:
    """Remove inf/-inf e aplica clip para valores extremos."""
    espectro = espectro.copy()
    espectro = np.clip(espectro, -CLIP_ABS, CLIP_ABS)
    espectro[~np.isfinite(espectro)] = np.nan
    return espectro


def interpolar(espectro: np.ndarray) -> np.ndarray:
    """Interpola NaNs usando pd.Series com extrapolação nas bordas."""
    espectro = espectro.copy()
    for i in range(espectro.shape[0]):
        linha = espectro[i]
        if not np.any(np.isnan(linha)):
            continue
        s = pd.Series(linha).interpolate(limit_direction="both")
        espectro[i] = s.to_numpy()
    espectro = np.nan_to_num(espectro, nan=0.0, posinf=0.0, neginf=0.0)
    return espectro


def preprocessar(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Aplica pré-processamento completo e retorna DataFrame + comprimentos de onda."""
    meta = [c for c in df.columns if not c.isdigit()]
    bandas = sorted(int(c) for c in df.columns if c.isdigit())
    bandas_str = [str(b) for b in bandas]

    mask = [(b >= LIMITE_INF and b <= LIMITE_SUP) for b in bandas]
    bandas_validas = [b for b, m in zip(bandas, mask) if m]
    bandas_validas_str = [str(b) for b in bandas_validas]
    w = np.array(bandas_validas, dtype=float)

    print(f"Bandas: {bandas_validas[0]}-{bandas_validas[-1]} nm "
          f"({len(bandas_validas)} bandas)")

    espectro = df[bandas_str].values.astype(float)
    espectro_valido = espectro[:, mask]

    espectro_limpo = limpar_valores_espurios(espectro_valido)

    espectro_corrigido = np.zeros_like(espectro_limpo)
    for i in range(espectro_limpo.shape[0]):
        s = pd.Series(espectro_limpo[i], index=w, dtype="float64")
        s = jump_correct(s, SPLICE_POINTS, REFERENCE_SEQ)
        espectro_corrigido[i] = s.to_numpy()

    espectro_interp = interpolar(espectro_corrigido)

    df_espectro = pd.DataFrame(espectro_interp, columns=bandas_validas_str, index=df.index)
    df_saida = pd.concat([df[meta], df_espectro], axis=1)

    return df_saida, w


def main() -> None:
    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    df_preprocessado, w = preprocessar(df)

    df_preprocessado.to_csv(SAIDA, sep=";", index=False)
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
