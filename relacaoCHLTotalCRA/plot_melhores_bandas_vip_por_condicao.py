#!/usr/bin/env python3
"""Relações da melhor banda VIP com CHLTotal e CRA, por condição hídrica.

Para cada melhor data do PLSR, entre as cinco bandas com maior VIP final é
escolhida aquela com maior |rho de Spearman| com o parâmetro. A unidade é a
média espectral matinal de cada bloco × genótipo × condição.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
SAIDA = Path(__file__).resolve().parent / "resultados"
CHAVES = ["bloco", "genotipo", "condicao"]
CORES = {"IRRIG": "#0072B2", "NIRRIG": "#E69F00"}
ROTULOS = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
ANALISES = {
    "chltotal": {
        "titulo": "CHLTotal em 25/02: relação com a melhor banda VIP",
        "ranking": ROOT / "analiseClorofilaTotal" / "resultados" / "25_02_bandas_vip_ge_1.csv",
        "coluna_rho": "rho_spearman_clorofila_total",
        "dia_espectral": "D04M",
        "coluna_parametro": "CHL TOTAL (25.02)",
        "rotulo_parametro": "CHLTotal observada",
        "arquivo": "chltotal_25_02_melhor_banda_vip_por_condicao",
    },
    "cra": {
        "titulo": "CRA em 02/03: relação com a melhor banda VIP",
        "ranking": ROOT / "analiseCRA" / "resultados" / "02_03_bandas_vip_ge_1.csv",
        "coluna_rho": "rho_spearman_cra",
        "dia_espectral": "D09M",
        "coluna_parametro": "CRA (02/03)",
        "rotulo_parametro": "CRA observada (%)",
        "arquivo": "cra_02_03_melhor_banda_vip_por_condicao",
    },
}


def dados_fisiologicos() -> pd.DataFrame:
    fisio = pd.read_excel(FISIO)
    fisio["genotipo"] = fisio["Genótipo"]
    fisio["condicao"] = fisio["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fisio["bloco"] = "B" + (fisio.groupby(["genotipo", "condicao"], sort=False).cumcount() + 1).astype(str)
    return fisio


def escolher_banda(config: dict) -> pd.Series:
    ranking = pd.read_csv(config["ranking"], sep=";").head(5).copy()
    return ranking.loc[ranking[config["coluna_rho"]].abs().idxmax()]


def dados_pareados(config: dict, banda: int) -> pd.DataFrame:
    espectros = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    coluna_banda = str(banda)
    medias = (espectros.loc[
        espectros["data_coleta"].eq(config["dia_espectral"]) & espectros["turno"].eq("manha"),
        CHAVES + [coluna_banda],
    ].groupby(CHAVES, as_index=False)[coluna_banda].mean())
    fisio = dados_fisiologicos()
    dados = fisio[CHAVES + [config["coluna_parametro"]]].merge(
        medias, on=CHAVES, how="inner", validate="one_to_one"
    ).dropna(subset=[config["coluna_parametro"], coluna_banda])
    return dados.rename(columns={config["coluna_parametro"]: "parametro", coluna_banda: "reflectancia"})


def plotar(nome: str, config: dict) -> dict:
    banda = escolher_banda(config)
    banda_nm = int(banda["banda_nm"])
    dados = dados_pareados(config, banda_nm)
    x = dados["reflectancia"].to_numpy(float)
    y = dados["parametro"].to_numpy(float)
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    predito = reg.predict(x.reshape(-1, 1))
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for condicao in ["IRRIG", "NIRRIG"]:
        parte = dados.loc[dados["condicao"].eq(condicao)]
        ax.scatter(parte["reflectancia"], parte["parametro"], s=70, color=CORES[condicao],
                   edgecolor="white", linewidth=.8, alpha=.92, label=ROTULOS[condicao])
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, reg.predict(xx.reshape(-1, 1)), "--", color="#333333", linewidth=1.3,
            label="Regressão linear")
    ax.text(.03, .97,
            f"Banda selecionada: {banda_nm} nm; VIP final = {banda['vip_modelo_final']:.3f}\n"
            f"ρ Spearman = {spearman.statistic:.3f}; p = {spearman.pvalue:.4f}; R² = {r2_score(y, predito):.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=14,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#666", "alpha": .94})
    ax.set(xlabel=f"Reflectância em {banda_nm} nm", ylabel=config["rotulo_parametro"], title=config["titulo"])
    ax.grid(alpha=.22)
    ax.legend(title="Condição", frameon=True, loc="lower right", fontsize=14, title_fontsize=14)
    fig.tight_layout()
    fig.savefig(SAIDA / f"{config['arquivo']}.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / f"{config['arquivo']}.pdf", bbox_inches="tight")
    plt.close(fig)
    dados.to_csv(SAIDA / f"{config['arquivo']}_dados.csv", sep=";", index=False, float_format="%.8f")
    return {
        "parametro": nome, "data": "25/02" if nome == "chltotal" else "02/03", "banda_nm": banda_nm,
        "vip_modelo_final": banda["vip_modelo_final"], "rho_spearman": spearman.statistic,
        "p_spearman": spearman.pvalue, "pearson_r": pearson.statistic, "r2_linear": r2_score(y, predito),
        "rmse_linear": mean_squared_error(y, predito) ** .5, "n": len(dados),
    }


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    resumo = pd.DataFrame([plotar(nome, config) for nome, config in ANALISES.items()])
    resumo.to_csv(SAIDA / "resumo_melhores_bandas_vip_por_condicao.csv", sep=";", index=False, float_format="%.6f")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
