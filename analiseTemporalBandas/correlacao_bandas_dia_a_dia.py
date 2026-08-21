#!/usr/bin/env python3
"""Correlacao dos perfis espectrais medios entre dias, por genotipo.

Cada perfil diario e a media das leituras da mesma combinacao
genotipo x condicao x dia, no turno da manha. A correlacao compara esses
perfis ao longo das 2.051 bandas de 400--2450 nm; portanto, responde quao
semelhante e o formato espectral de dois dias. IRRIG e NIRRIG permanecem
separados, para que uma diferenca de condicao nao seja interpretada como efeito
temporal.

Sao reportados Pearson (semelhanca linear) e Spearman (semelhanca monotona).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402

SAIDA = ROOT / "dataset_gerado" / "correlacao_dias"
TURNO = "manha"
ESTAGIO = "normalizado"
METODOS = ("Pearson", "Spearman")


def correlacionar(perfis: dict[str, np.ndarray], metodo: str) -> pd.DataFrame:
    """Monta uma matriz simetrica de correlacoes entre perfis diarios."""
    dias = sorted(perfis)
    matriz = pd.DataFrame(np.eye(len(dias)), index=dias, columns=dias)
    for d1, d2 in combinations(dias, 2):
        if metodo == "Pearson":
            r, _ = pearsonr(perfis[d1], perfis[d2])
        else:
            r, _ = spearmanr(perfis[d1], perfis[d2])
        matriz.loc[d1, d2] = matriz.loc[d2, d1] = r
    return matriz


def pares_longos(matriz: pd.DataFrame, genotipo: str, condicao: str, metodo: str) -> list[dict]:
    """Converte o triangulo superior da matriz para uma tabela de pares."""
    return [
        {
            "genotipo": genotipo,
            "condicao": condicao,
            "metodo": metodo,
            "dia_1": d1,
            "dia_2": d2,
            "correlacao": matriz.loc[d1, d2],
        }
        for d1, d2 in combinations(matriz.index, 2)
    ]


def plotar(matrizes: dict[tuple[str, str], pd.DataFrame]) -> None:
    genotipos = sorted({g for g, _ in matrizes})
    condicoes = ("IRRIG", "NIRRIG")
    fig, axes = plt.subplots(len(genotipos), len(condicoes), figsize=(10, 12),
                             constrained_layout=True)
    im = None
    for i, genotipo in enumerate(genotipos):
        for j, condicao in enumerate(condicoes):
            ax = axes[i, j]
            matriz = matrizes[(genotipo, condicao)]
            im = ax.imshow(matriz, vmin=0.75, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(matriz.columns)), matriz.columns, rotation=45)
            ax.set_yticks(range(len(matriz.index)), matriz.index)
            ax.set_title(f"{genotipo} — {condicao}", fontweight="bold")
            for y in range(len(matriz.index)):
                for x in range(len(matriz.columns)):
                    valor = matriz.iloc[y, x]
                    ax.text(x, y, f"{valor:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if valor < 0.88 else "black")
    fig.colorbar(im, ax=axes, shrink=0.85, label="r de Pearson")
    fig.suptitle("Correlacao dia a dia dos perfis espectrais medios\n"
                 "2.051 bandas (400–2450 nm), turno da manha", fontweight="bold")
    fig.savefig(SAIDA / "heatmap_correlacao_pearson_dias_genotipo_condicao.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    meta, espectro, _ = carregar(ESTAGIO, turno=TURNO)
    dias = sorted(meta["dia"].unique())
    genotipos = sorted(meta["genotipo"].unique())
    condicoes = ("IRRIG", "NIRRIG")
    resultados = []
    matrizes_pearson: dict[tuple[str, str], pd.DataFrame] = {}

    for genotipo in genotipos:
        for condicao in condicoes:
            base = (meta["genotipo"].eq(genotipo) & meta["condicao"].eq(condicao)).to_numpy()
            perfis = {dia: espectro[base & meta["dia"].eq(dia).to_numpy()].mean(axis=0)
                      for dia in dias}
            for metodo in METODOS:
                matriz = correlacionar(perfis, metodo)
                matriz.to_csv(SAIDA / f"correlacao_{metodo.lower()}_{genotipo}_{condicao}.csv",
                              sep=";", float_format="%.6f")
                resultados.extend(pares_longos(matriz, genotipo, condicao, metodo))
                if metodo == "Pearson":
                    matrizes_pearson[(genotipo, condicao)] = matriz

    pares = pd.DataFrame(resultados)
    pares.to_csv(SAIDA / "correlacoes_dias_genotipo_condicao.csv", sep=";",
                 index=False, float_format="%.6f")
    resumo = (pares.groupby(["genotipo", "condicao", "metodo"], as_index=False)
              .agg(correlacao_media=("correlacao", "mean"),
                   correlacao_minima=("correlacao", "min"),
                   correlacao_maxima=("correlacao", "max")))
    resumo.to_csv(SAIDA / "resumo_correlacoes_dias.csv", sep=";", index=False,
                  float_format="%.6f")
    plotar(matrizes_pearson)
    print(resumo.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nResultados salvos em: {SAIDA}")


if __name__ == "__main__":
    main()
