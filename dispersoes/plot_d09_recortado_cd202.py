#!/usr/bin/env python3
"""Gera o painel isolado D09 – Recortado para o genótipo CD202."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402
from plot_dispersao_dia_genotipo import (  # noqa: E402
    CONDICOES,
    legenda_figura,
    plot_celula,
    resumo_banda,
)


GENOTIPO = "CD202"
DIA = "D09"
CHAVE_RECORTADO = "recortado"
ROTULO = "Recortado (recorte + jump correction)"


def main() -> None:
    meta, estagios, w = carregar_estagios(turno="manha")
    mask_base = ((meta["genotipo"] == GENOTIPO) & (meta["dia"] == DIA)).to_numpy()
    resumo = {}
    for condicao in CONDICOES:
        mask = mask_base & (meta["condicao"] == condicao).to_numpy()
        if mask.any():
            resumo[condicao] = resumo_banda(estagios[CHAVE_RECORTADO][mask])

    if not resumo:
        raise SystemExit(f"Nenhuma leitura encontrada para {GENOTIPO}, {DIA}.")

    # Mantém a escala vertical da figura multipainel original para CD202.
    mask_genotipo = (meta["genotipo"] == GENOTIPO).to_numpy()
    valores = estagios[CHAVE_RECORTADO][mask_genotipo]
    y_min, y_max = valores.min(), valores.max()
    folga = 0.05 * (y_max - y_min)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    ns = ", ".join(f"{c} n={resumo[c]['n']}" for c in CONDICOES if c in resumo)
    plot_celula(ax, w, resumo, f"{DIA} - {ROTULO}  ({ns})", "Reflectância")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=14)
    ax.set_ylabel("Reflectância", fontsize=14)
    ax.set_ylim(y_min - folga, y_max + folga)
    legenda_figura(fig)
    fig.suptitle(
        f"Genótipo {GENOTIPO} – dispersão espectral\n"
        "Turno da manhã, IRRIG e NIRRIG separados",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.91])
    saida = ROOT / "dispersao_dia_CD202_D09_recortado.png"
    fig.savefig(saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(saida)


if __name__ == "__main__":
    main()
