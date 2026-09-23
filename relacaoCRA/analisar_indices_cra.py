#!/usr/bin/env python3
"""Relaciona índices espectrais de água/clorofila ao CRA por parcela.

As oito leituras espectrais matinais de cada parcela são reduzidas à média do
bloco. Essa média é então pareada ao CRA da mesma combinação de bloco,
genótipo e condição. São tratadas separadamente as três datas fisiológicas,
pois uma correlação agrupada entre datas pode refletir somente a mudança
temporal simultânea das duas variáveis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ENTRADA_ESPECTRAL = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
ENTRADA_FISIO = ROOT / "ParametrosFisiologicos.xlsx"
SAIDA = ROOT / "relacaoCRA" / "resultados"
CHAVES = ["bloco", "genotipo", "condicao"]

# Data fisiológica: (coluna de CRA, coleta espectral matinal correspondente).
COLETAS = {
    "24/02": ("CRA (24/02)", "D03M"),
    "02/03": ("CRA (02/03)", "D09M"),
    "03/03": ("CRA (03/03)", "D10M"),
}
CORES = {"24/02": "#287a8e", "02/03": "#e8902f", "03/03": "#8c4f9d"}

# As razões são calculadas sobre reflectância bruta média, preservando a escala
# física. SRWI e NDWI usam o par NIR–SWIR 860/1240 nm disponível no sensor.
INDICES = {
    "WBI": {
        "formula": "R900 / R970",
        "descricao": "Índice de água foliar",
        "calcular": lambda d: d["900"] / d["970"],
    },
    "SRWI": {
        "formula": "R860 / R1240",
        "descricao": "Índice de água de razão simples",
        "calcular": lambda d: d["860"] / d["1240"],
    },
    "NDWI": {
        "formula": "(R860 − R1240) / (R860 + R1240)",
        "descricao": "Índice normalizado de água",
        "calcular": lambda d: (d["860"] - d["1240"]) / (d["860"] + d["1240"]),
    },
    "Clorofila a (razão)": {
        "formula": "R430 / R622",
        "descricao": "Proxy espectral de clorofila a",
        "calcular": lambda d: d["430"] / d["622"],
    },
    "Clorofila b (razão)": {
        "formula": "R453 / R642",
        "descricao": "Proxy espectral de clorofila b",
        "calcular": lambda d: d["453"] / d["642"],
    },
    "MCARI (clorofila total)": {
        "formula": "[(R700 − R670) − 0,2 × (R700 − R550)] × (R700 / R670)",
        "descricao": "Índice de absorção de clorofila total",
        "calcular": lambda d: ((d["700"] - d["670"]) - 0.2 * (d["700"] - d["550"])) * (d["700"] / d["670"]),
    },
}


def preparar_dados() -> pd.DataFrame:
    """Calcula índices nas médias de bloco e os pareia ao CRA."""
    espectros = pd.read_csv(ENTRADA_ESPECTRAL, sep=";", decimal=",")
    bandas = ["430", "453", "550", "622", "642", "670", "700", "860", "900", "970", "1240"]
    medias = (espectros.loc[espectros["turno"].eq("manha"), CHAVES + ["data_coleta", *bandas]]
              .groupby(CHAVES + ["data_coleta"], as_index=False)[bandas].mean())

    fisio = pd.read_excel(ENTRADA_FISIO)
    fisio["genotipo"] = fisio["Genótipo"]
    fisio["condicao"] = fisio["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    # A ordem das quatro repetições em cada combinação identifica B1--B4.
    fisio["bloco"] = (fisio.groupby(["genotipo", "condicao"], sort=False).cumcount() + 1)
    fisio["bloco"] = "B" + fisio["bloco"].astype(str)

    partes = []
    for data, (coluna_cra, dia_espectral) in COLETAS.items():
        x = medias.loc[medias["data_coleta"].eq(dia_espectral)].copy()
        y = fisio[CHAVES + [coluna_cra]].rename(columns={coluna_cra: "CRA"})
        pareado = y.merge(x, on=CHAVES, how="inner", validate="one_to_one").dropna(subset=["CRA"])
        if len(pareado) < 8:
            raise ValueError(f"{data}: pareamento insuficiente ({len(pareado)} parcelas).")
        for nome, config in INDICES.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                pareado[nome] = config["calcular"](pareado)
        pareado["data_cra"] = data
        pareado["dia_espectral"] = dia_espectral
        partes.append(pareado)
    dados = pd.concat(partes, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return dados.dropna(subset=["CRA", *INDICES])


def correlacoes(dados: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    grupos = [(data, parte) for data, parte in dados.groupby("data_cra", sort=False)]
    grupos.append(("Todas as datas (descritivo)", dados))
    for data, parte in grupos:
        for indice, config in INDICES.items():
            x, y = parte[indice].to_numpy(float), parte["CRA"].to_numpy(float)
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_rho, spearman_p = spearmanr(x, y)
            ajuste = linregress(x, y)
            linhas.append({
                "data_cra": data, "indice": indice, "formula": config["formula"],
                "descricao": config["descricao"], "n": len(parte),
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_rho": spearman_rho, "spearman_p": spearman_p,
                "r2_linear": ajuste.rvalue ** 2, "inclinacao": ajuste.slope,
            })
    return pd.DataFrame(linhas)


def grafico_dispersao(dados: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, eixos = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, (indice, config) in zip(eixos.flat, INDICES.items()):
        for data, parte in dados.groupby("data_cra", sort=False):
            x, y = parte[indice], parte["CRA"]
            ajuste = linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.scatter(x, y, s=43, color=CORES[data], edgecolor="white", linewidth=.6, alpha=.9, label=data)
            ax.plot(xx, ajuste.intercept + ajuste.slope * xx, color=CORES[data], linewidth=1.4, alpha=.8)
        ax.set_title(indice, fontweight="bold")
        ax.set_xlabel(config["formula"], fontsize=9)
        ax.set_ylabel("CRA (%)")
        ax.grid(alpha=.25)
    handles, labels = eixos.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Data do CRA", loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Relação entre índices espectrais e conteúdo relativo de água", fontsize=15, fontweight="bold")
    fig.text(.5, .925, "Cada ponto = média das leituras matinais de uma parcela; linhas = ajuste linear por data", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .90))
    fig.savefig(SAIDA / "dispersao_indices_vs_cra.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "dispersao_indices_vs_cra.pdf", bbox_inches="tight")
    plt.close(fig)


def grafico_heatmap(tabela: pd.DataFrame) -> None:
    ordem = list(COLETAS) + ["Todas as datas (descritivo)"]
    matriz = (tabela.pivot(index="indice", columns="data_cra", values="spearman_rho")
              .reindex(index=list(INDICES), columns=ordem))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    imagem = ax.imshow(matriz, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(ordem)), ordem)
    ax.set_yticks(range(len(matriz.index)), matriz.index)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz.iat[i, j]
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="white" if abs(valor) > .55 else "black", fontweight="bold")
    fig.colorbar(imagem, ax=ax, label="ρ de Spearman com CRA")
    ax.set_title("Correlação por data entre cada índice e o CRA")
    fig.tight_layout()
    fig.savefig(SAIDA / "heatmap_spearman_indices_cra.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "heatmap_spearman_indices_cra.pdf", bbox_inches="tight")
    plt.close(fig)


def relatorio(tabela: pd.DataFrame, dados: pd.DataFrame) -> None:
    linhas = [
        "# Índices espectrais × CRA", "",
        "Unidade experimental: média das oito leituras espectrais matinais por bloco × genótipo × condição, pareada ao CRA da mesma parcela.",
        "Datas pareadas: D03M–24/02, D09M–02/03 e D10M–03/03. O conjunto possui " + str(len(dados)) + " parcelas válidas.", "",
        "Os resultados por data são a evidência principal. A linha com todas as datas é apenas descritiva, pois pode incorporar o efeito simultâneo do tempo em CRA e índice.", "",
        "| Data CRA | Índice | n | ρ Spearman | p Spearman | r Pearson | p Pearson | R² linear |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in tabela.iterrows():
        linhas.append(f"| {r['data_cra']} | {r['indice']} | {r['n']} | {r['spearman_rho']:.3f} | {r['spearman_p']:.4f} | {r['pearson_r']:.3f} | {r['pearson_p']:.4f} | {r['r2_linear']:.3f} |")
    linhas += ["", "## Fórmulas", ""]
    for nome, config in INDICES.items():
        linhas.append(f"- **{nome}:** `{config['formula']}` — {config['descricao']}.")
    (SAIDA / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    dados = preparar_dados()
    tabela = correlacoes(dados)
    dados[CHAVES + ["data_cra", "dia_espectral", "CRA", *INDICES]].to_csv(
        SAIDA / "dados_pareados_indices_cra.csv", sep=";", index=False, float_format="%.8f"
    )
    tabela.to_csv(SAIDA / "correlacoes_indices_cra.csv", sep=";", index=False, float_format="%.6f")
    grafico_dispersao(dados)
    grafico_heatmap(tabela)
    relatorio(tabela, dados)
    print(f"Análise concluída: {len(dados)} parcelas e {len(INDICES)} índices em {SAIDA}")


if __name__ == "__main__":
    main()
