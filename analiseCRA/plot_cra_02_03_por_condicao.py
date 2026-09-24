#!/usr/bin/env python3
"""CRA observado × predito em 02/03, distinguindo condição hídrica."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "analiseCRA" / "resultados"
ARQUIVO = RESULTADOS / "02_03_predicoes_cv.csv"
OBSERVADO = "CRA (02/03)"
CORES = {"IRRIG": "#0072B2", "NIRRIG": "#E69F00"}
ROTULOS = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}


def main() -> None:
    dados = pd.read_csv(ARQUIVO, sep=";").dropna(subset=[OBSERVADO, "predito_cv"])
    minimo = min(dados[OBSERVADO].min(), dados["predito_cv"].min())
    maximo = max(dados[OBSERVADO].max(), dados["predito_cv"].max())
    margem = (maximo - minimo) * .06

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    linhas = []
    for condicao in ["IRRIG", "NIRRIG"]:
        parte = dados.loc[dados["condicao"].eq(condicao)]
        ax.scatter(parte[OBSERVADO], parte["predito_cv"], s=70, color=CORES[condicao],
                   edgecolor="white", linewidth=.8, alpha=.92, label=ROTULOS[condicao])
        linhas.append({
            "condicao": condicao, "rotulo": ROTULOS[condicao], "n": len(parte),
            "r2_cv": r2_score(parte[OBSERVADO], parte["predito_cv"]),
            "rmse_cv": mean_squared_error(parte[OBSERVADO], parte["predito_cv"]) ** .5,
        })

    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem],
            "--", color="#333333", linewidth=1.2, label="1:1")
    r2_total = r2_score(dados[OBSERVADO], dados["predito_cv"])
    rmse_total = mean_squared_error(dados[OBSERVADO], dados["predito_cv"]) ** .5
    ax.text(.03, .97, f"Todos os dados: R² CV = {r2_total:.3f}\nRMSE = {rmse_total:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=15,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#666", "alpha": .94})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel="CRA observado (%)", ylabel="CRA predito (CV por bloco, %)",
           title="CRA em 02/03: observado × predito por condição hídrica")
    ax.grid(alpha=.22)
    ax.legend(title="Condição", frameon=True, loc="lower right", fontsize=15, title_fontsize=15)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "02_03_cra_observado_x_predito_por_condicao.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTADOS / "02_03_cra_observado_x_predito_por_condicao.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(linhas).to_csv(RESULTADOS / "02_03_cra_metricas_por_condicao.csv", sep=";", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
