#!/usr/bin/env python3
"""Painel comparativo de espectros recortados vs normalizados, por genotipo.

Mesma leitura de `plot_preprocessamento.py`, com o genotipo aberto: onde
aquele painel junta os tres materiais numa media so por dia, este separa.
Grid 7x6:

- 7 linhas: dias de coleta (D02 a D10)
- 6 colunas: os tres genotipos, cada um com recortado e normalizado (SNV)
- Cada subplot: medias espectrais de IRRIG e NIRRIG

O que a separacao mostra e por que ela importa: a distancia entre as duas
curvas no estagio recortado e em boa parte um efeito de escala -- a planta
estressada reflete mais em todo o espectro. O SNV tira essa escala, e o que
sobra do lado direito e diferenca de *forma*. Os genotipos nao chegam la do
mesmo jeito, e e isso que o painel deixa comparar lado a lado.

Os dois estagios saem de `carregar_estagios`, o mesmo pipeline em memoria que
todos os modulos de analise usam, e apenas o turno da manha entra.

Uso:
    python plot_preprocessamento_genotipo.py
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402
from plot_preprocessamento import COR_IRRIG, COR_NIRRIG  # noqa: E402

PASTA_SAIDA = ROOT.parent / "dispersoes"
SAIDA = PASTA_SAIDA / "preprocessamento_comparativo_genotipo.png"

TURNO = "manha"

CONDICOES = ["IRRIG", "NIRRIG"]
COR_CONDICAO = {"IRRIG": COR_IRRIG, "NIRRIG": COR_NIRRIG}

ESTAGIOS = [
    ("recortado", "Recortado", "Reflectancia"),
    ("normalizado", "Normalizado (SNV)", "Reflectancia (SNV)"),
]


def calcular_medias(
    meta: pd.DataFrame,
    espectro: np.ndarray,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Media espectral de cada combinacao (dia, genotipo, condicao)."""
    medias = {}
    for dia in sorted(meta["dia"].unique()):
        for genotipo in sorted(meta["genotipo"].unique()):
            for condicao in CONDICOES:
                mask = (
                    (meta["dia"] == dia)
                    & (meta["genotipo"] == genotipo)
                    & (meta["condicao"] == condicao)
                ).to_numpy()
                if mask.any():
                    medias[(dia, genotipo, condicao)] = espectro[mask].mean(axis=0)
    return medias


def plot_celula(
    ax: plt.Axes,
    w: np.ndarray,
    medias: dict[tuple[str, str, str], np.ndarray],
    dia: str,
    genotipo: str,
    titulo: str,
    unidade: str,
    mostrar_legenda: bool,
) -> None:
    """Um subplot: IRRIG e NIRRIG de um dia, genotipo e estagio."""
    presentes = []
    for condicao in CONDICOES:
        chave = (dia, genotipo, condicao)
        if chave not in medias:
            continue
        ax.plot(w, medias[chave], color=COR_CONDICAO[condicao], linewidth=1.5,
                label=condicao)
        presentes.append(condicao)

    ax.set_title(titulo, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    ax.set_ylabel(unidade, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    if mostrar_legenda and presentes:
        ax.legend(fontsize=7, loc="upper right")


def gerar_painel(
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
) -> None:
    """Gera o painel comparativo completo."""
    print("\nCalculando medias por dia, genotipo e condicao...")
    medias = {chave: calcular_medias(meta, estagios[chave])
              for chave, _, _ in ESTAGIOS}

    dias = sorted(meta["dia"].unique())
    genotipos = sorted(meta["genotipo"].unique())
    print(f"  Dias: {', '.join(dias)}")
    print(f"  Genotipos: {', '.join(genotipos)}")

    # Escala compartilhada por coluna: sem isso cada subplot escolhe seu
    # proprio limite e a comparacao entre dias -- que e o ponto da figura --
    # vira ilusao de otica.
    limites = {}
    for chave, _, _ in ESTAGIOS:
        valores = np.concatenate([v for v in medias[chave].values()])
        folga = 0.05 * (valores.max() - valores.min())
        limites[chave] = (valores.min() - folga, valores.max() + folga)

    plt.style.use("seaborn-v0_8-whitegrid")
    n_colunas = len(genotipos) * len(ESTAGIOS)
    fig, axes = plt.subplots(len(dias), n_colunas, figsize=(4.6 * n_colunas, 21))

    print(f"\nGerando {len(dias) * n_colunas} subplots...")
    for i, dia in enumerate(dias):
        for j, genotipo in enumerate(genotipos):
            for k, (chave, rotulo, unidade) in enumerate(ESTAGIOS):
                coluna = j * len(ESTAGIOS) + k
                ax = axes[i, coluna]
                plot_celula(
                    ax, w, medias[chave], dia, genotipo,
                    f"{dia} | {genotipo} - {rotulo}", unidade,
                    mostrar_legenda=(i == 0 and coluna == 0),
                )
                ax.set_ylim(*limites[chave])

    fig.suptitle(
        "Recorte + jump correction vs Savitzky-Golay + SNV, por dia, genotipo "
        "e condicao\n"
        "Medias espectrais de IRRIG e NIRRIG, turno da manha  -  escala do eixo "
        "y compartilhada por estagio",
        fontsize=15,
        fontweight="bold",
        y=0.997,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(SAIDA, dpi=150, bbox_inches="tight")
    print(f"\nPainel salvo em: {SAIDA}")
    print(f"  Layout: {len(dias)} linhas (dias) x {n_colunas} colunas "
          f"({len(genotipos)} genotipos x {len(ESTAGIOS)} estagios)")


def main() -> None:
    print("Gerando painel comparativo de preprocessamento por genotipo...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")
    gerar_painel(meta, estagios, w)
    print("Concluido.")


if __name__ == "__main__":
    main()
