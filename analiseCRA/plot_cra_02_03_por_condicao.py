#!/usr/bin/env python3
"""Painéis CRA observado × predito por condição, para todas as coletas."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "analiseCRA" / "resultados"
CORES = {"IRRIG": "#0072B2", "NIRRIG": "#E69F00"}
ROTULOS = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
COLETAS = {
    "24_02": ("24/02", "CRA (24/02)"),
    "02_03": ("02/03", "CRA (02/03)"),
    "03_03": ("03/03", "CRA (03/03)"),
}


def plotar(codigo: str, data: str, observado: str) -> None:
    dados = pd.read_csv(RESULTADOS / f"{codigo}_predicoes_cv.csv", sep=";").dropna(subset=[observado, "predito_cv"])
    minimo = min(dados[observado].min(), dados["predito_cv"].min())
    maximo = max(dados[observado].max(), dados["predito_cv"].max())
    margem = (maximo - minimo) * .06
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    linhas = []
    for condicao in ["IRRIG", "NIRRIG"]:
        parte = dados.loc[dados["condicao"].eq(condicao)]
        ax.scatter(parte[observado], parte["predito_cv"], s=70, color=CORES[condicao], edgecolor="white",
                   linewidth=.8, alpha=.92, label=ROTULOS[condicao])
        linhas.append({"data": data, "condicao": condicao, "rotulo": ROTULOS[condicao], "n": len(parte),
                       "r2_cv": r2_score(parte[observado], parte["predito_cv"]),
                       "rmse_cv": mean_squared_error(parte[observado], parte["predito_cv"]) ** .5})
    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem], "--", color="#333333", linewidth=1.2, label="1:1")
    ax.text(.03, .97, f"R² CV = {r2_score(dados[observado], dados['predito_cv']):.3f}\nRMSE = {mean_squared_error(dados[observado], dados['predito_cv']) ** .5:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=15,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#666", "alpha": .94})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel="CRA observado (%)", ylabel="CRA predito (CV por bloco, %)",
           title=f"CRA em {data}: observado × predito por condição hídrica")
    ax.xaxis.label.set_fontsize(18); ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(18); ax.yaxis.label.set_fontweight("bold")
    ax.grid(alpha=.22); ax.legend(title="Condição", frameon=True, loc="lower right", fontsize=15, title_fontsize=15)
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"{codigo}_cra_observado_x_predito_por_condicao.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTADOS / f"{codigo}_cra_observado_x_predito_por_condicao.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(linhas).to_csv(RESULTADOS / f"{codigo}_cra_metricas_por_condicao.csv", sep=";", index=False, float_format="%.6f")


if __name__ == "__main__":
    for codigo, (data, observado) in COLETAS.items():
        plotar(codigo, data, observado)
