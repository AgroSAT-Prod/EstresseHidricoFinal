#!/usr/bin/env python3
"""Suavização espectral usando filtro Savitzky-Golay.

O filtro Savitzky-Golay ajusta polinômios locais por mínimos quadrados,
preservando características espectrais (picos, vales) enquanto remove
ruído de alta frequência. É o método padrão para suavização de dados
espectrais de reflectância.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_400_2450.csv"
SAIDA = ROOT.parent / "dataset" / "Unificada13052026_suavizado.csv"

WINDOW_LENGTH = 11
POLYORDER = 2
MODE = "interp"


def savitzky_golay(
    espectro: np.ndarray,
    window_length: int = WINDOW_LENGTH,
    polyorder: int = POLYORDER,
    mode: str = MODE,
) -> np.ndarray:
    """Aplica filtro Savitzky-Golay em cada amostra do espectro.

    Args:
        espectro: Array 2D (n_amostras, n_bandas)
        window_length: Tamanho da janela (deve ser ímpar)
        polyorder: Ordem do polinômio
        mode: Modo de tratamento das bordas ('interp', 'mirror', etc.)

    Returns:
        Espectro suavizado com mesma dimensão
    """
    if window_length % 2 == 0:
        window_length += 1

    min_window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    window_length = max(window_length, min_window)

    espectro_suavizado = np.zeros_like(espectro)

    for i in range(espectro.shape[0]):
        linha = espectro[i]
        if len(linha) >= window_length:
            espectro_suavizado[i] = savgol_filter(
                linha, window_length, polyorder, mode=mode
            )
        else:
            espectro_suavizado[i] = linha

    return espectro_suavizado


def main() -> None:
    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    meta = [c for c in df.columns if not c.isdigit()]
    bandas = sorted(int(c) for c in df.columns if c.isdigit())
    bandas_str = [str(b) for b in bandas]

    espectro = df[bandas_str].values.astype(float)

    print(f"Aplicando Savitzky-Golay (window={WINDOW_LENGTH}, "
          f"poly={POLYORDER}, mode='{MODE}')...")

    espectro_suavizado = savitzky_golay(espectro)

    df_suavizado = df.copy()
    df_suavizado[bandas_str] = espectro_suavizado

    df_suavizado.to_csv(SAIDA, sep=";", index=False)
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
