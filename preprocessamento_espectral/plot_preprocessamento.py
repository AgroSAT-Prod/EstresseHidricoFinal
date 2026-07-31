#!/usr/bin/env python3
"""Painel comparativo de espectros brutos vs normalizados por dia e condicao.

Gera uma unica figura PNG com grid 7x2:
- 7 linhas: dias de coleta (D02 a D10)
- 2 colunas: bruto (esquerda) vs normalizado (direita)
- Cada subplot: medias espectrais de IRRIG (azul) e NIRRIG (vermelho)

Uso:
    python plot_preprocessamento.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT.parent / "dataset"
BRUTO_CSV = DATASET_DIR / "Unificada13052026_Limpa.csv"
NORMALIZADO_CSV = DATASET_DIR / "Unificada13052026_normalizado.csv"

SAIDA = ROOT / "preprocessamento_comparativo.png"

LIMITE_INF = 400
LIMITE_SUP = 2450

COR_IRRIG = "#2ca02c"
COR_NIRRIG = "#d62728"


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Carrega datasets bruto e normalizado, aplica recorte espectral."""
    print("Carregando dados brutos...")
    df_bruto = pd.read_csv(BRUTO_CSV, sep=";", decimal=",")
    df_bruto.columns = [c.strip() for c in df_bruto.columns]
    
    print("Carregando dados normalizados...")
    df_norm = pd.read_csv(NORMALIZADO_CSV, sep=";", decimal=",")
    df_norm.columns = [c.strip() for c in df_norm.columns]
    
    bandas_bruto = sorted(int(c) for c in df_bruto.columns if c.isdigit())
    bandas_norm = sorted(int(c) for c in df_norm.columns if c.isdigit())
    
    mask_bruto = [(b >= LIMITE_INF and b <= LIMITE_SUP) for b in bandas_bruto]
    mask_norm = [(b >= LIMITE_INF and b <= LIMITE_SUP) for b in bandas_norm]
    
    bandas_validas_bruto = [b for b, m in zip(bandas_bruto, mask_bruto) if m]
    bandas_validas_norm = [b for b, m in zip(bandas_norm, mask_norm) if m]
    
    w_bruto = np.array(bandas_validas_bruto, dtype=float)
    w_norm = np.array(bandas_validas_norm, dtype=float)
    
    print(f"  Bruto: {len(df_bruto)} amostras, {len(w_bruto)} bandas ({w_bruto.min():.0f}-{w_bruto.max():.0f} nm)")
    print(f"  Normalizado: {len(df_norm)} amostras, {len(w_norm)} bandas ({w_norm.min():.0f}-{w_norm.max():.0f} nm)")
    
    return df_bruto, df_norm, w_bruto, w_norm


def extrair_dia(data_coleta: str) -> str:
    """Extrai o dia (D02, D03, etc) da string de data_coleta."""
    import re
    match = re.match(r"^(D\d+)", str(data_coleta))
    return match.group(1) if match else str(data_coleta)


def calcular_medias_por_dia_condicao(
    df: pd.DataFrame,
    band_cols: list[str],
) -> dict[tuple[str, str], np.ndarray]:
    """Calcula media espectral para cada combinacao (dia, condicao)."""
    df = df.copy()
    df["dia"] = df["data_coleta"].apply(extrair_dia)
    
    bandas = df[band_cols].apply(pd.to_numeric, errors="coerce")
    
    medias = {}
    for dia in sorted(df["dia"].unique()):
        for condicao in ["IRRIG", "NIRRIG"]:
            mask = (df["dia"] == dia) & (df["condicao"] == condicao)
            if mask.any():
                medias[(dia, condicao)] = bandas.loc[mask].mean().to_numpy()
    
    return medias


def plot_subplot(
    ax: plt.Axes,
    w: np.ndarray,
    medias: dict[tuple[str, str], np.ndarray],
    dia: str,
    titulo: str,
    mostrar_legenda: bool = True,
) -> None:
    """Plota um subplot com IRRIG e NIRRIG."""
    if (dia, "IRRIG") in medias:
        ax.plot(w, medias[(dia, "IRRIG")], color=COR_IRRIG, linewidth=1.5, label="IRRIG")
    if (dia, "NIRRIG") in medias:
        ax.plot(w, medias[(dia, "NIRRIG")], color=COR_NIRRIG, linewidth=1.5, label="NIRRIG")
    
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    ax.set_ylabel("Reflectancia", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    
    if mostrar_legenda and (dia, "IRRIG") in medias:
        ax.legend(fontsize=7, loc="upper right")


def gerar_painel(
    df_bruto: pd.DataFrame,
    df_norm: pd.DataFrame,
    w_bruto: np.ndarray,
    w_norm: np.ndarray,
) -> None:
    """Gera o painel comparativo completo."""
    bandas_bruto_str = [str(int(b)) for b in w_bruto]
    bandas_norm_str = [str(int(b)) for b in w_norm]
    
    print("\nCalculando medias por dia e condicao...")
    medias_bruto = calcular_medias_por_dia_condicao(df_bruto, bandas_bruto_str)
    medias_norm = calcular_medias_por_dia_condicao(df_norm, bandas_norm_str)
    
    dias = sorted(set(d for d, c in medias_bruto.keys()))
    print(f"  Dias encontrados: {', '.join(dias)}")
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(dias), 2, figsize=(16, 21))
    
    print(f"\nGerando {len(dias) * 2} subplots...")
    for i, dia in enumerate(dias):
        ax_bruto = axes[i, 0]
        ax_norm = axes[i, 1]
        
        plot_subplot(ax_bruto, w_bruto, medias_bruto, dia, f"{dia} - Bruto", mostrar_legenda=(i == 0))
        plot_subplot(ax_norm, w_norm, medias_norm, dia, f"{dia} - Normalizado", mostrar_legenda=(i == 0))
    
    fig.text(0.25, 0.96, "Dados Brutos", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.75, 0.96, "Dados Normalizados", ha="center", fontsize=12, fontweight="bold")
    
    fig.suptitle(
        "Comparacao de espectros brutos vs normalizados por dia e condicao\n"
        "IRRIG (verde) vs NIRRIG (vermelho) - medias espectrais",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(SAIDA, dpi=300, bbox_inches="tight")
    print(f"\nPainel salvo em: {SAIDA}")
    print(f"  Resolucao: 300 DPI")
    print(f"  Tamanho: 16x21 pol")
    print(f"  Layout: {len(dias)} linhas (dias) x 2 colunas (bruto vs normalizado)")


def main() -> None:
    print("Gerando painel comparativo de preprocessamento...\n")
    df_bruto, df_norm, w_bruto, w_norm = carregar_dados()
    gerar_painel(df_bruto, df_norm, w_bruto, w_norm)
    print("Concluido.")


if __name__ == "__main__":
    main()
