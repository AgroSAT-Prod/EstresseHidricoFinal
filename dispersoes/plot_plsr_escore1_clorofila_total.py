#!/usr/bin/env python3
"""Dispersão real do escore PLSR para clorofila total.

Reconstitui o modelo final de ``plsr_chltotal_normalizado_spearman_p005``.
O modelo selecionado pela validação GroupKFold tem uma componente latente;
por isso, a figura relaciona o escore dessa componente com a clorofila total
observada, em vez de exibir um eixo inexistente de segunda componente.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
BANDAS_FINAIS = (
    ROOT
    / "analiseClorofilaTotal"
    / "resultados"
    / "todos_dias_normalizado_spearman_p005_vip_nao_redundante"
    / "bandas_finais.csv"
)
SAIDA = ROOT / "dispersoes" / "plsr_escore1_clorofila_total.png"

MAPA = {
    "23/02": ("CHL TOTAL (23/02)", "D02M"),
    "24/02": ("CHL TOTAL (24/02)", "D03M"),
    "25/02": ("CHL TOTAL (25.02)", "D04M"),
    "26/02": ("CHL TOTAL (26/02)", "D05M"),
    "27/02": ("CHL TOTAL (27/02)", "D06M"),
    "02/03": ("CHL TOTAL (02/03)", "D09M"),
    "03/03": ("CHL TOTAL (03/03)", "D10M"),
}
CORES = {"EMB48": "#16733a", "CD202": "#2878c8", "BR16": "#d85a25"}


def pre_processar_normalizar(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Replica o pré-processamento do projeto sem depender de SciPy.

    Aplica recorte 400--2450 nm, correção das emendas de detectores, filtro
    Savitzky--Golay (janela 11, polinômio 2) e SNV. A implementação do filtro
    usa os mesmos coeficientes locais; os extremos, que não entram nas bandas
    finais (721 e 1318 nm), não são usados no gráfico.
    """
    bandas = np.array(sorted(int(coluna) for coluna in df.columns if coluna.isdigit()))
    bandas = bandas[(bandas >= 400) & (bandas <= 2450)]
    colunas = bandas.astype(str).tolist()
    espectro = df[colunas].to_numpy(float)
    espectro = np.clip(espectro, -1e4, 1e4)
    espectro[~np.isfinite(espectro)] = np.nan
    espectro = pd.DataFrame(espectro).interpolate(axis=1, limit_direction="both").fillna(0.0).to_numpy().copy()

    # Correção dos degraus nas emendas em 1000 e 1800 nm, usando SWIR1 como referência.
    i1000 = np.searchsorted(bandas, 1000, side="right")
    i1800 = np.searchsorted(bandas, 1800, side="right")
    espectro[:, i1800:] += (espectro[:, i1800 - 1] - espectro[:, i1800])[:, None]
    espectro[:, :i1000] += (espectro[:, i1000] - espectro[:, i1000 - 1])[:, None]

    posicoes = np.arange(-5, 6, dtype=float)
    matriz = np.column_stack([posicoes**grau for grau in range(3)])
    pesos = np.linalg.pinv(matriz)[0]
    suavizado = espectro.copy()
    for indice in range(5, espectro.shape[1] - 5):
        suavizado[:, indice] = espectro[:, indice - 5 : indice + 6] @ pesos
    media = suavizado.mean(axis=1, keepdims=True)
    desvio = suavizado.std(axis=1, keepdims=True)
    normalizado = np.divide(suavizado - media, desvio, out=np.zeros_like(suavizado), where=desvio > 0)
    meta = df[[coluna for coluna in df.columns if not coluna.isdigit()]].copy()
    return meta, normalizado, bandas


def escore_pls_um_componente(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula o escore X da única componente PLSR do modelo final."""
    x_padronizado = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    y_padronizado = (y - y.mean()) / y.std(ddof=1)
    pesos = x_padronizado.T @ y_padronizado
    pesos /= np.linalg.norm(pesos)
    return x_padronizado @ pesos


def medias_por_bloco(meta: pd.DataFrame, espectro: np.ndarray, dia: str) -> tuple[pd.DataFrame, np.ndarray]:
    chaves = ["bloco", "genotipo", "condicao"]
    mascara = meta.data_coleta.eq(dia).to_numpy()
    meta_dia = meta.loc[mascara, chaves].reset_index(drop=True)
    espectro_dia = espectro[mascara]
    indices = meta_dia.groupby(chaves, sort=True).indices
    linhas, medias = [], []
    for valores, indice in indices.items():
        linhas.append(dict(zip(chaves, valores)))
        medias.append(espectro_dia[indice].mean(axis=0))
    return pd.DataFrame(linhas), np.asarray(medias)


def carregar_dados() -> tuple[pd.DataFrame, list[str]]:
    fisiologicos = pd.read_excel(FISIO)
    fisiologicos["genotipo"] = fisiologicos["Genótipo"]
    fisiologicos["condicao"] = fisiologicos["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fisiologicos["bloco"] = (
        fisiologicos.groupby(["genotipo", "condicao"], sort=False)
        .cumcount()
        .map(lambda indice: f"B{indice + 1}")
    )

    espectros_brutos = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    espectros_brutos.columns = [coluna.strip() for coluna in espectros_brutos.columns]
    espectros_brutos = espectros_brutos.loc[espectros_brutos["turno"].eq("manha")].reset_index(drop=True)
    meta, espectro, bandas = pre_processar_normalizar(espectros_brutos)
    chaves = ["bloco", "genotipo", "condicao"]
    partes = []
    for data, (alvo, dia) in MAPA.items():
        metad, xd = medias_por_bloco(meta, espectro, dia)
        espectro_df = pd.DataFrame(xd, columns=bandas.astype(int).astype(str))
        medias = pd.concat([metad, espectro_df], axis=1)
        parte = (
            fisiologicos[chaves + [alvo]]
            .merge(medias, on=chaves, how="inner", validate="one_to_one")
            .dropna(subset=[alvo])
            .rename(columns={alvo: "chl_total"})
        )
        parte.insert(0, "data", data)
        partes.append(parte)
    dados = pd.concat(partes, ignore_index=True)
    bandas_finais = pd.read_csv(BANDAS_FINAIS, sep=";")["banda_nm"].astype(str).tolist()
    return dados, bandas_finais


def main() -> None:
    dados, bandas_finais = carregar_dados()
    x = dados[bandas_finais].to_numpy(float)
    y = dados["chl_total"].to_numpy(float)
    dados = dados.assign(escore_pls1=escore_pls_um_componente(x, y))

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for genotipo, cor in CORES.items():
        for condicao, preenchido in (("IRRIG", True), ("NIRRIG", False)):
            grupo = dados[(dados.genotipo == genotipo) & (dados.condicao == condicao)]
            ax.scatter(
                grupo.escore_pls1,
                grupo.chl_total,
                s=48,
                facecolors=cor if preenchido else "none",
                edgecolors=cor,
                linewidths=1.25,
                alpha=0.82,
            )

    coeficientes = np.polyfit(dados.escore_pls1, dados.chl_total, deg=1)
    eixo_x = np.linspace(dados.escore_pls1.min(), dados.escore_pls1.max(), 100)
    ax.plot(eixo_x, np.polyval(coeficientes, eixo_x), color="#535353", linewidth=1.2, linestyle="--")
    ax.axvline(0, color="#b9b9b9", linewidth=0.8)
    ax.grid(alpha=0.18)
    ax.set(
        xlabel="Escore do componente latente 1 (PLSR)",
        ylabel="Clorofila total observada",
        title="PLSR: escore latente e clorofila total",
    )

    legenda_genotipo = [
        Line2D([0], [0], marker="o", color="none", label=nome, markerfacecolor=cor, markeredgecolor=cor, markersize=8)
        for nome, cor in CORES.items()
    ]
    legenda_regime = [
        Line2D([0], [0], marker="o", color="black", label="Irrigado", markerfacecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="black", label="Não irrigado", markerfacecolor="none", markersize=8),
    ]
    primeira = ax.legend(handles=legenda_genotipo, title="Genótipo", loc="upper left")
    ax.add_artist(primeira)
    ax.legend(handles=legenda_regime, title="Regime hídrico", loc="lower right")
    fig.tight_layout()
    fig.savefig(SAIDA, dpi=300, bbox_inches="tight")
    print(f"Figura salva em: {SAIDA}")
    print(f"Amostras: {len(dados)}; bandas finais: {', '.join(bandas_finais)}")


if __name__ == "__main__":
    main()
