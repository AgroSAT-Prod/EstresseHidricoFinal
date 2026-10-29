#!/usr/bin/env python3
"""Gera graficos de CHLTotal observado x predito por data e condicao hidrica."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "analiseClorofilaTotal" / "resultados"
DATAS = ("23_02", "24_02", "25_02", "26_02", "27_02", "02_03", "03_03")
CORES = {"IRRIG": "#2474b5", "NIRRIG": "#d95f02"}
ROTULOS = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}


def coluna_observada(tabela: pd.DataFrame) -> str:
    """Localiza a coluna de CHLTotal, que muda levemente entre as datas."""
    return next(coluna for coluna in tabela if coluna.upper().startswith("CHL TOTAL"))


def main() -> None:
    todas_predicoes = []
    top5 = []

    for nome_data in DATAS:
        tabela = pd.read_csv(RESULTADOS / f"{nome_data}_predicoes_cv.csv", sep=";")
        observado = coluna_observada(tabela)
        dia = nome_data.replace("_", "/")
        dados = tabela[["condicao", observado, "predito_cv"]].rename(
            columns={observado: "chltotal_observado", "predito_cv": "chltotal_predito_cv"}
        ).dropna()
        dados.insert(0, "dia", dia)
        todas_predicoes.append(dados)

        bandas = pd.read_csv(RESULTADOS / f"{nome_data}_bandas_vip_ge_1.csv", sep=";")
        top5.append(bandas.nlargest(5, "vip_modelo_final").assign(dia=dia))

    predicoes = pd.concat(todas_predicoes, ignore_index=True)
    predicoes.to_csv(RESULTADOS / "chltotal_observado_x_predito_por_dia.csv", sep=";", index=False)
    pd.concat(top5, ignore_index=True).to_csv(
        RESULTADOS / "chltotal_top5_vip_por_dia.csv", sep=";", index=False
    )

    metricas = []
    for dia in predicoes["dia"].drop_duplicates():
        parte = predicoes[predicoes["dia"].eq(dia)]
        minimo = min(parte.chltotal_observado.min(), parte.chltotal_predito_cv.min())
        maximo = max(parte.chltotal_observado.max(), parte.chltotal_predito_cv.max())
        margem = (maximo - minimo) * .06

        fig, ax = plt.subplots(figsize=(6.7, 5.7))
        for condicao in ("IRRIG", "NIRRIG"):
            grupo = parte[parte.condicao.eq(condicao)]
            ax.scatter(grupo.chltotal_observado, grupo.chltotal_predito_cv, s=66,
                       color=CORES[condicao], edgecolor="white", linewidth=.8, alpha=.92,
                       label=ROTULOS[condicao])
        ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem],
                "--", color="#333333", linewidth=1.2, label="1:1")
        r2 = r2_score(parte.chltotal_observado, parte.chltotal_predito_cv)
        rmse = mean_squared_error(parte.chltotal_observado, parte.chltotal_predito_cv) ** .5
        ax.text(.03, .97, f"R² CV = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
                va="top", ha="left", bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
                "edgecolor": "#666", "alpha": .94})
        ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
               xlabel="CHLTotal observada", ylabel="CHLTotal predita (CV por bloco)",
               title=f"CHLTotal: observado × predito — {dia}")
        ax.grid(alpha=.22)
        ax.legend(title="Condição", frameon=True, loc="lower right")
        fig.tight_layout()
        arquivo = f"chltotal_observado_x_predito_{dia.replace('/', '_')}"
        fig.savefig(RESULTADOS / f"{arquivo}.png", dpi=300, bbox_inches="tight")
        fig.savefig(RESULTADOS / f"{arquivo}.pdf", bbox_inches="tight")
        plt.close(fig)
        metricas.append({"dia": dia, "n": len(parte), "r2_cv_bloco": r2, "rmse_cv_bloco": rmse})

    pd.DataFrame(metricas).to_csv(
        RESULTADOS / "chltotal_observado_x_predito_metricas.csv", sep=";", index=False,
        float_format="%.6f"
    )


if __name__ == "__main__":
    main()
