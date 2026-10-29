#!/usr/bin/env python3
"""Gera graficos observado x predito de E e gs, separados por condicao hidrica."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
PASTA_E = ROOT / "analiseFisiologica" / "resultados_e_todos_dias_vip_global_plsr_diario"
PASTA_GS = ROOT / "analiseFisiologica" / "resultados_gs_25_02_n24_por_bloco"
DATAS_E = ("23_02", "24_02", "25_02", "26_02", "27_02", "02_03", "03_03")
CORES = {"IRR": "#2474b5", "NIR": "#d95f02"}
ROTULOS = {"IRR": "Irrigado", "NIR": "Não irrigado"}


def gerar_grafico(dados, observado, predito, variavel, dia, destino):
    minimo = min(dados[observado].min(), dados[predito].min())
    maximo = max(dados[observado].max(), dados[predito].max())
    margem = (maximo - minimo) * .06
    r2 = r2_score(dados[observado], dados[predito])
    rmse = mean_squared_error(dados[observado], dados[predito]) ** .5

    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    for condicao in ("IRR", "NIR"):
        grupo = dados[dados.condicao.eq(condicao)]
        ax.scatter(grupo[observado], grupo[predito], s=66, color=CORES[condicao],
                   edgecolor="white", linewidth=.8, alpha=.92, label=ROTULOS[condicao])
    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem],
            "--", color="#333333", linewidth=1.2, label="1:1")
    ax.text(.03, .97, f"R² LOO = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
            va="top", ha="left", bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
            "edgecolor": "#666", "alpha": .94})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel=f"{variavel} observada", ylabel=f"{variavel} predita (LOO)",
           title=f"{variavel}: observado × predito — {dia}")
    ax.grid(alpha=.22)
    ax.legend(title="Condição", frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(destino.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(destino.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"dia": dia, "n": len(dados), "r2_loo": r2, "rmse_loo": rmse}


def main():
    metricas_e = []
    predicoes_e = []
    for nome_data in DATAS_E:
        dados = pd.read_csv(PASTA_E / f"{nome_data}_predicoes.csv", sep=";").dropna(subset=["e", "predito_loo"])
        dia = nome_data.replace("_", "/")
        metricas_e.append(gerar_grafico(dados, "e", "predito_loo", "E", dia,
                           PASTA_E / f"e_observado_x_predito_{nome_data}"))
        predicoes_e.append(dados.assign(dia=dia))
    pd.DataFrame(metricas_e).to_csv(PASTA_E / "e_observado_x_predito_metricas.csv",
                                    sep=";", index=False, float_format="%.6f")
    pd.concat(predicoes_e, ignore_index=True).to_csv(PASTA_E / "e_observado_x_predito_por_dia.csv",
                                                      sep=";", index=False)

    dados_gs = pd.read_csv(PASTA_GS / "dados_pareados_n24_predicoes.csv", sep=";").dropna(subset=["gs", "predito_loo"])
    metricas_gs = gerar_grafico(dados_gs, "gs", "predito_loo", "Condutância estomática", "25/02",
                                PASTA_GS / "gs_observado_x_predito_25_02")
    pd.DataFrame([metricas_gs]).to_csv(PASTA_GS / "gs_observado_x_predito_metricas.csv",
                                       sep=";", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
