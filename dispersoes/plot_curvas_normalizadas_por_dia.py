#!/usr/bin/env python3
"""Gera curvas espectrais SNV por dia, com genotipos agrupados.

Para cada dia de coleta, compara as medias das leituras matinais irrigadas e
nao irrigadas. Os tres genotipos e todos os blocos sao reunidos em cada
condicao. O processamento reproduz o fluxo do projeto: recorte/correcao,
suavizacao Savitzky-Golay e normalizacao SNV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402


PASTA_SAIDA = ROOT / "dispersoes" / "curvas_espectrais_normalizadas_por_dia"
TURNO = "manha"
CONDICOES = (
    ("IRRIG", "Irrigado", "#1976D2"),
    ("NIRRIG", "Nao irrigado", "#D32F2F"),
)


def ordenar_dias(dia: str) -> int:
    """Extrai o numero do codigo D02, D03, ... para ordenar cronologicamente."""
    return int(dia[1:])


def main() -> None:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    meta, espectro, bandas = carregar("normalizado", turno=TURNO)
    dias = sorted(meta["dia"].dropna().unique(), key=ordenar_dias)

    plt.style.use("seaborn-v0_8-whitegrid")
    for dia in dias:
        mascara_dia = meta["dia"].eq(dia).to_numpy()
        fig, ax = plt.subplots(figsize=(11, 6.5))

        for condicao, rotulo, cor in CONDICOES:
            mascara = mascara_dia & meta["condicao"].eq(condicao).to_numpy()
            dados = espectro[mascara]
            if len(dados) == 0:
                raise ValueError(f"Sem dados para {dia} / {condicao}.")

            media = np.nanmean(dados, axis=0)
            erro_padrao = np.nanstd(dados, axis=0, ddof=1) / np.sqrt(len(dados))
            ax.plot(bandas, media, color=cor, linewidth=2.3, label=f"{rotulo} (n={len(dados)})")
            ax.fill_between(bandas, media - erro_padrao, media + erro_padrao, color=cor, alpha=0.16)

        ax.axhline(0, color="#555555", linewidth=0.7, alpha=0.55)
        ax.set_title(
            f"Curvas espectrais normalizadas - {dia}\n"
            "Genotipos CD202, BR16 e EMB48 agrupados; leituras matinais",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Comprimento de onda (nm)", fontsize=12)
        ax.set_ylabel("Reflectancia normalizada (SNV)", fontsize=12)
        ax.grid(True, alpha=0.28)
        ax.legend(title="Condicao hidrica", frameon=True)
        fig.tight_layout()

        saida = PASTA_SAIDA / f"curvas_espectrais_normalizadas_{dia}.png"
        fig.savefig(saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(saida)


if __name__ == "__main__":
    main()
