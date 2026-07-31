#!/usr/bin/env python3
"""Painel comparativo de espectros recortados vs normalizados por dia e condicao.

Gera uma unica figura PNG com grid 7x2:
- 7 linhas: dias de coleta (D02 a D10)
- 2 colunas: recortado (esquerda) vs normalizado (direita)
- Cada subplot: medias espectrais de IRRIG e NIRRIG

Os dois estagios saem de `carregar_estagios`, o mesmo pipeline em memoria que
todos os modulos de analise usam. A versao anterior lia um CSV intermediario
(`Unificada13052026_normalizado.csv`) que nao existe no repositorio, entao a
figura so podia ser gerada a partir de um arquivo local nao versionado.

Apenas o turno da manha entra, como no resto do projeto.

Uso:
    python plot_preprocessamento.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402

SAIDA = ROOT / "preprocessamento_comparativo.png"

TURNO = "manha"

# Par validado contra deuteranopia, protanopia e tritanopia (dE 33.6 em visao
# normal, 26.5 no pior caso de CVD). Substitui o verde/vermelho anterior, que
# e exatamente o par que daltonismo vermelho-verde nao distingue.
COR_IRRIG = "#2a78d6"
COR_NIRRIG = "#eb6834"


def calcular_medias_por_dia_condicao(
    meta: pd.DataFrame,
    espectro: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    """Calcula media espectral para cada combinacao (dia, condicao)."""
    medias = {}
    for dia in sorted(meta["dia"].unique()):
        for condicao in ["IRRIG", "NIRRIG"]:
            mask = ((meta["dia"] == dia) & (meta["condicao"] == condicao)).to_numpy()
            if mask.any():
                medias[(dia, condicao)] = espectro[mask].mean(axis=0)
    return medias


def plot_subplot(
    ax: plt.Axes,
    w: np.ndarray,
    medias: dict[tuple[str, str], np.ndarray],
    dia: str,
    titulo: str,
    mostrar_legenda: bool = True,
    unidade: str = "Reflectancia",
) -> None:
    """Plota um subplot com IRRIG e NIRRIG."""
    if (dia, "IRRIG") in medias:
        ax.plot(w, medias[(dia, "IRRIG")], color=COR_IRRIG, linewidth=1.5, label="IRRIG")
    if (dia, "NIRRIG") in medias:
        ax.plot(w, medias[(dia, "NIRRIG")], color=COR_NIRRIG, linewidth=1.5, label="NIRRIG")

    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    ax.set_ylabel(unidade, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    
    if mostrar_legenda and (dia, "IRRIG") in medias:
        ax.legend(fontsize=7, loc="upper right")


def gerar_painel(
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
) -> None:
    """Gera o painel comparativo completo."""
    print("\nCalculando medias por dia e condicao...")
    medias_recorte = calcular_medias_por_dia_condicao(meta, estagios["recortado"])
    medias_norm = calcular_medias_por_dia_condicao(meta, estagios["normalizado"])

    dias = sorted(set(d for d, c in medias_recorte.keys()))
    print(f"  Dias encontrados: {', '.join(dias)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(dias), 2, figsize=(16, 21))

    print(f"\nGerando {len(dias) * 2} subplots...")
    for i, dia in enumerate(dias):
        plot_subplot(axes[i, 0], w, medias_recorte, dia,
                     f"{dia} - Recortado", mostrar_legenda=(i == 0))
        plot_subplot(axes[i, 1], w, medias_norm, dia,
                     f"{dia} - Normalizado (SNV)", mostrar_legenda=(i == 0),
                     unidade="Reflectancia (SNV)")

    fig.suptitle(
        "Recorte + jump correction vs Savitzky-Golay + SNV, por dia e condicao\n"
        "Medias espectrais de IRRIG e NIRRIG, turno da manha",
        fontsize=14,
        fontweight="bold",
        y=0.997,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(SAIDA, dpi=300, bbox_inches="tight")
    print(f"\nPainel salvo em: {SAIDA}")
    print("  Resolucao: 300 DPI")
    print("  Tamanho: 16x21 pol")
    print(f"  Layout: {len(dias)} linhas (dias) x 2 colunas (recortado vs normalizado)")


def main() -> None:
    print("Gerando painel comparativo de preprocessamento...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")
    gerar_painel(meta, estagios, w)
    print("Concluido.")


if __name__ == "__main__":
    main()
