"""Normalização espectral usando Standard Normal Variate (SNV).

SNV remove efeitos de espalhamento de luz e variações de caminho óptico,
aplicando centralização e escalonamento individual para cada espectro:

    x_snv = (x - mean(x)) / std(x)

Método padrão para correção de efeitos multiplicativos em espectroscopia
NIR/SWIR de material vegetal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_suavizado.csv"
SAIDA = ROOT.parent / "dataset" / "Unificada13052026_normalizado.csv"


def snv(espectro: np.ndarray) -> np.ndarray:
    """Aplica Standard Normal Variate em cada amostra do espectro.

    Args:
        espectro: Array 2D (n_amostras, n_bandas)

    Returns:
        Espectro normalizado com mesma dimensão
    """
    espectro_snv = np.zeros_like(espectro)

    for i in range(espectro.shape[0]):
        linha = espectro[i]
        media = np.nanmean(linha)
        desvio = np.nanstd(linha)

        if desvio > 0 and np.isfinite(desvio):
            espectro_snv[i] = (linha - media) / desvio
        else:
            espectro_snv[i] = linha - media

    return espectro_snv


def main() -> None:
    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    meta = [c for c in df.columns if not c.isdigit()]
    bandas = sorted(int(c) for c in df.columns if c.isdigit())
    bandas_str = [str(b) for b in bandas]

    espectro = df[bandas_str].values.astype(float)

    print(f"Aplicando SNV em {espectro.shape[0]} amostras...")

    espectro_normalizado = snv(espectro)

    df_normalizado = df.copy()
    df_normalizado[bandas_str] = espectro_normalizado

    df_normalizado.to_csv(SAIDA, sep=";", index=False)
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()