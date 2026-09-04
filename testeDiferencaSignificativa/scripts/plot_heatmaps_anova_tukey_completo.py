#!/usr/bin/env python3
"""Painel único dos heatmaps ANOVA + Tukey, incluindo efeito da irrigação."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as exc:
    raise SystemExit("matplotlib nao esta instalado. Execute: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "resultados" / "dataset_gerado"
SAIDA_PNG = ROOT.parent / "resultados" / "heatmaps_anova_tukey_completo.png"
SAIDA_PDF = ROOT.parent / "resultados" / "heatmaps_anova_tukey_completo.pdf"
N_BANDAS = 2051
CONDICOES = ["IRRIG", "NIRRIG"]
PARES = [("BR16", "CD202"), ("BR16", "EMB48"), ("CD202", "EMB48")]
CMAP = LinearSegmentedColormap.from_list("anovatukey", ["#f7f8fa", "#8fc1c4", "#087f8c"])
COR_CONDICAO = {"IRRIG": "#2878b5", "NIRRIG": "#d96d31"}


def formatar(n: int, percentual: float) -> str:
    return f"{n:,}\n{percentual:.1f}%".replace(",", ".")


def desenhar(ax: plt.Axes, valores: np.ndarray, contagens: np.ndarray,
             dias: list[str], rotulos: list[str], titulo: str, separadores: list[float] = []) -> None:
    ax.imshow(valores, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            cor = "white" if valores[i, j] >= 55 else "#17212b"
            ax.text(j, i, formatar(int(contagens[i, j]), valores[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=cor)
    for y in separadores:
        ax.axhline(y, color="white", linewidth=2.2)
    ax.set_xticks(range(len(dias)), dias, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(rotulos)), rotulos, fontsize=11.5)
    ax.set_title(titulo, fontsize=14.5, fontweight="bold", pad=11)


def buscar(df: pd.DataFrame, dia: str, **filtros) -> pd.Series:
    mascara = df.dia.eq(dia)
    for coluna, valor in filtros.items():
        mascara &= df[coluna].eq(valor)
    resultado = df[mascara]
    if len(resultado) != 1:
        raise ValueError(f"Resultado ausente ou duplicado: dia={dia}, filtros={filtros}")
    return resultado.iloc[0]


def contar_significativas_tukey(df: pd.DataFrame, dia: str, **filtros) -> int:
    """Conta as bandas significativas de um contraste presente linha a linha."""
    mascara = df.dia.eq(dia)
    for coluna, valor in filtros.items():
        mascara &= df[coluna].eq(valor)
    resultado = df[mascara]
    if resultado.empty:
        raise ValueError(f"Contraste Tukey ausente: dia={dia}, filtros={filtros}")
    return int(resultado.significativo_tukey.sum())


def gerar() -> None:
    resumo_anova = pd.read_csv(DATA / "anova_tukey_genotipos_resumo.csv", sep=";")
    resumo_mesma = pd.read_csv(DATA / "anova_tukey_genotipos_resumo_pares.csv", sep=";")
    # Este arquivo contém os 15 pares de Tukey entre as seis células
    # (3 genótipos × 2 condições), inclusive Irrigado × Não irrigado
    # dentro de cada genótipo.
    tukey_seis_celulas = pd.read_csv(DATA / "tukey_seis_celulas_por_dia.csv", sep=";")
    dias = sorted(resumo_anova.dia.unique())
    if len(dias) != 7:
        raise ValueError("Esperados sete dias de avaliacao.")

    valores_anova = np.array([[100 * buscar(resumo_anova, dia, condicao=cond).prop_bandas_sig_anova
                                for dia in dias] for cond in CONDICOES])
    contagens_anova = np.array([[buscar(resumo_anova, dia, condicao=cond).bandas_sig_anova
                                  for dia in dias] for cond in CONDICOES], dtype=int)

    linhas_mesma = [(cond, a, b) for cond in CONDICOES for a, b in PARES]
    valores_mesma = np.array([[100 * buscar(resumo_mesma, dia, condicao=cond, genotipo_a=a,
                                              genotipo_b=b).bandas_sig_tukey / N_BANDAS
                               for dia in dias] for cond, a, b in linhas_mesma])
    contagens_mesma = np.array([[buscar(resumo_mesma, dia, condicao=cond, genotipo_a=a,
                                         genotipo_b=b).bandas_sig_tukey
                                for dia in dias] for cond, a, b in linhas_mesma], dtype=int)

    linhas_irrigacao = [(genotipo, "IRRIG", genotipo, "NIRRIG")
                         for genotipo in ["BR16", "CD202", "EMB48"]]
    valores_irrigacao = np.array([[100 * contar_significativas_tukey(
                                      tukey_seis_celulas, dia, genotipo_a=a, condicao_a=cond_a,
                                      genotipo_b=b, condicao_b=cond_b) / N_BANDAS
                                   for dia in dias] for a, cond_a, b, cond_b in linhas_irrigacao])
    contagens_irrigacao = np.array([[contar_significativas_tukey(
                                        tukey_seis_celulas, dia, genotipo_a=a, condicao_a=cond_a,
                                        genotipo_b=b, condicao_b=cond_b)
                                    for dia in dias] for a, cond_a, b, cond_b in linhas_irrigacao], dtype=int)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18.5, 15.5), layout="constrained")
    grade = fig.add_gridspec(3, 1, height_ratios=[1.5, 3.0, 3.0], hspace=0.36)
    ax_anova = fig.add_subplot(grade[0])
    ax_mesma = fig.add_subplot(grade[1])
    ax_irrigacao = fig.add_subplot(grade[2])

    desenhar(ax_anova, valores_anova, contagens_anova, dias, CONDICOES,
             "A. ANOVA entre genótipos dentro de cada condição")
    for tick, cond in zip(ax_anova.get_yticklabels(), CONDICOES):
        tick.set_color(COR_CONDICAO[cond]); tick.set_fontweight("bold")
    desenhar(ax_mesma, valores_mesma, contagens_mesma, dias,
             [f"{cond} | {a} × {b}" for cond, a, b in linhas_mesma],
             "B. Tukey HSD: genótipos comparados dentro da mesma condição", [2.5])
    desenhar(ax_irrigacao, valores_irrigacao, contagens_irrigacao, dias,
             [f"{a}: Irrigado × Não irrigado" for a, _, _, _ in linhas_irrigacao],
             "C. Tukey HSD: irrigado × não irrigado dentro de cada genótipo", [0.5, 1.5])

    fig.suptitle("Diferenças espectrais entre genótipos e condições, dia a dia",
                 fontsize=19, fontweight="bold")
    barra = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 100)),
                         ax=[ax_anova, ax_mesma, ax_irrigacao], location="right", pad=0.01, fraction=0.025)
    barra.set_label("Bandas significativas (%)", fontsize=13)
    barra.ax.tick_params(labelsize=12)
    fig.savefig(SAIDA_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG salvo em: {SAIDA_PNG}")
    print(f"PDF salvo em: {SAIDA_PDF}")


if __name__ == "__main__":
    gerar()
