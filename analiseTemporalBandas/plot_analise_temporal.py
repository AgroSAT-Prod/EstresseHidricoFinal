#!/usr/bin/env python3
"""Painel visual da analise temporal das bandas espectrais.

Gera uma unica figura PNG com multiplos subplots mostrando:
- Heatmaps de separacao temporal (matriz de pares de dias) por grupo
- Perfil espectral do efeito temporal (efeito ao longo do espectro)
- Metricas comparativas por grupo

Uso:
    python plot_analise_temporal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent

SAIDA_DIR = ROOT / "dataset_gerado"
SAIDA = ROOT / "analise_temporal_painel.png"

DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
GRUPOS = ["todas", "IRRIG", "NIRRIG"]
CORES_GRUPOS = {"todas": "#404040", "IRRIG": "#245f73", "NIRRIG": "#d95f02"}


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os resultados da analise temporal."""
    temporal = pd.read_csv(SAIDA_DIR / "temporal_por_banda.csv", sep=";")
    separacao = pd.read_csv(SAIDA_DIR / "separacao_temporal.csv", sep=";")
    resumo = pd.read_csv(SAIDA_DIR / "temporal_resumo.csv", sep=";")
    
    print("Dados carregados:")
    print(f"  temporal_por_banda: {len(temporal)} linhas")
    print(f"  separacao_temporal: {len(separacao)} linhas")
    print(f"  temporal_resumo: {len(resumo)} linhas")
    
    return temporal, separacao, resumo


def plot_heatmaps_separacao(
    axes: np.ndarray,
    separacao: pd.DataFrame,
) -> None:
    """Plota heatmaps de separacao temporal para cada grupo."""
    for idx, grupo in enumerate(GRUPOS):
        ax = axes[idx]
        sub = separacao[separacao["grupo"] == grupo].copy()
        
        matriz = np.full((len(DIAS), len(DIAS)), np.nan)
        
        for _, row in sub.iterrows():
            i = DIAS.index(row["dia_a"])
            j = DIAS.index(row["dia_b"])
            matriz[i, j] = row["prop_sig"]
            matriz[j, i] = row["prop_sig"]
        
        im = ax.imshow(matriz, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DIAS)))
        ax.set_yticks(range(len(DIAS)))
        ax.set_xticklabels(DIAS, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(DIAS, fontsize=8)
        ax.set_title(f"{grupo}", fontweight="bold", fontsize=10)
        
        for i in range(len(DIAS)):
            for j in range(len(DIAS)):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="gray")
                elif not np.isnan(matriz[i, j]):
                    valor = matriz[i, j]
                    cor = "white" if valor > 0.6 else "black"
                    ax.text(j, i, f"{valor:.0%}", ha="center", va="center", fontsize=6, color=cor)
        
        if idx == 0:
            ax.set_ylabel("Dia", fontsize=9)
        ax.set_xlabel("Dia", fontsize=9)
    
    cbar = plt.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.04, label="Proporcao de bandas significativas")


def plot_perfil_espectral(
    ax: plt.Axes,
    temporal: pd.DataFrame,
) -> None:
    """Plota o efeito temporal ao longo do espectro para cada grupo."""
    for grupo in GRUPOS:
        sub = temporal[temporal["grupo"] == grupo].copy()
        sub = sub.sort_values("banda_nm")
        
        ax.plot(
            sub["banda_nm"],
            sub["efeito_tempo"],
            color=CORES_GRUPOS[grupo],
            linewidth=1.2,
            alpha=0.8,
            label=grupo,
        )
        
        sig = sub[sub["significativa"]]
        if len(sig) > 0:
            ax.fill_between(
                sig["banda_nm"],
                0,
                sig["efeito_tempo"],
                color=CORES_GRUPOS[grupo],
                alpha=0.15,
            )
    
    ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Limiar 0.1")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel("Efeito temporal (eta² ou W)", fontsize=9)
    ax.set_title("Efeito temporal ao longo do espectro", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(400, 2500)
    ax.set_ylim(0, 1)


def plot_metricas_comparativas(
    ax: plt.Axes,
    resumo: pd.DataFrame,
) -> None:
    """Plota metricas comparativas por grupo."""
    resumo = resumo.set_index("grupo").loc[GRUPOS].reset_index()
    
    x = np.arange(len(GRUPOS))
    largura = 0.35
    
    barras1 = ax.bar(
        x - largura / 2,
        resumo["prop_bandas_sig"] * 100,
        largura,
        label="Bandas significativas (%)",
        color=[CORES_GRUPOS[g] for g in GRUPOS],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    
    ax2 = ax.twinx()
    barras2 = ax2.bar(
        x + largura / 2,
        resumo["efeito_mediano"],
        largura,
        label="Efeito mediano",
        color=[CORES_GRUPOS[g] for g in GRUPOS],
        alpha=0.4,
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
    )
    
    ax.set_xlabel("Grupo", fontsize=9)
    ax.set_ylabel("Proporcao de bandas significativas (%)", fontsize=9, color=CORES_GRUPOS["todas"])
    ax2.set_ylabel("Efeito temporal mediano (eta² ou W)", fontsize=9, color=CORES_GRUPOS["IRRIG"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(GRUPOS, fontsize=9)
    ax.set_title("Metricas comparativas por grupo", fontweight="bold", fontsize=10)
    
    ax.tick_params(axis="y", labelcolor=CORES_GRUPOS["todas"])
    ax2.tick_params(axis="y", labelcolor=CORES_GRUPOS["IRRIG"])
    
    linhas1, rotulos1 = ax.get_legend_handles_labels()
    linhas2, rotulos2 = ax2.get_legend_handles_labels()
    ax.legend(linhas1 + linhas2, rotulos1 + rotulos2, fontsize=8, loc="upper left")
    
    ax.grid(True, axis="y", alpha=0.3)
    
    for barra in barras1:
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f"{altura:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    
    for barra in barras2:
        altura = barra.get_height()
        ax2.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f"{altura:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_top_bandas(
    ax: plt.Axes,
    temporal: pd.DataFrame,
) -> None:
    """Plota as top 10 bandas mais sensiveis por grupo."""
    top_n = 10
    bandas = []
    posicoes = []
    efeitos = []
    grupos = []
    
    for grupo in GRUPOS:
        sub = temporal[temporal["grupo"] == grupo].copy()
        sub = sub.sort_values("efeito_tempo", ascending=False).head(top_n)
        bandas.extend(sub["banda_nm"].tolist())
        posicoes.extend(range(len(sub)))
        efeitos.extend(sub["efeito_tempo"].tolist())
        grupos.extend([grupo] * len(sub))
    
    df_plot = pd.DataFrame({
        "banda": bandas,
        "posicao": posicoes,
        "efeito": efeitos,
        "grupo": grupos,
    })
    
    for grupo in GRUPOS:
        sub = df_plot[df_plot["grupo"] == grupo]
        offset = GRUPOS.index(grupo) * 0.25 - 0.25
        ax.barh(
            sub["posicao"] + offset,
            sub["efeito"],
            height=0.25,
            label=grupo,
            color=CORES_GRUPOS[grupo],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
    
    ax.set_yticks(range(top_n))
    top_bandas_labels = df_plot[df_plot["grupo"] == "todas"].sort_values("posicao")["banda"].tolist()
    ax.set_yticklabels([f"{b} nm" for b in top_bandas_labels], fontsize=8)
    ax.set_xlabel("Efeito temporal (eta² ou W)", fontsize=9)
    ax.set_title(f"Top {top_n} bandas mais sensiveis", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()


def gerar_painel(temporal: pd.DataFrame, separacao: pd.DataFrame, resumo: pd.DataFrame) -> None:
    """Gera o painel completo."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    fig = plt.figure(figsize=(18, 12))
    
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1],
        hspace=0.35,
        wspace=0.35,
    )
    
    ax_heat1 = fig.add_subplot(gs[0, 0])
    ax_heat2 = fig.add_subplot(gs[0, 1])
    ax_heat3 = fig.add_subplot(gs[0, 2])
    ax_perfil = fig.add_subplot(gs[1, :])
    ax_metricas = fig.add_subplot(gs[2, 0])
    ax_top = fig.add_subplot(gs[2, 1:])
    
    plot_heatmaps_separacao(np.array([ax_heat1, ax_heat2, ax_heat3]), separacao)
    plot_perfil_espectral(ax_perfil, temporal)
    plot_metricas_comparativas(ax_metricas, resumo)
    plot_top_bandas(ax_top, temporal)
    
    fig.suptitle(
        "Analise temporal das bandas espectrais - Evolucao espectral ao longo do experimento",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
    plt.savefig(SAIDA, dpi=300, bbox_inches="tight")
    print(f"\nPainel salvo em: {SAIDA}")
    print(f"  Resolucao: 300 DPI")
    print(f"  Tamanho: 18x12 pol")


def main() -> None:
    print("Gerando painel de analise temporal...\n")
    temporal, separacao, resumo = carregar_dados()
    gerar_painel(temporal, separacao, resumo)
    print("Concluido.")


if __name__ == "__main__":
    main()
