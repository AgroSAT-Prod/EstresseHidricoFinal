#!/usr/bin/env python3
"""Gráficos por data para o PLSR/iPLS de CRA.

As cores identificam os genótipos; quadrados representam IRR e triângulos,
NIR. As bandas são as cinco selecionadas no resultado iPLS informado.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

import clorofila_total_spearman_plsr as base
from comparar_selecao_bandas_cra import ler_cra


RAIZ = Path(__file__).resolve().parent
ENTRADA = RAIZ / "resultados_cra_spearman_ipls10"
SAIDA = ENTRADA / "graficos_por_dia"
CORES = {"BR16": "#2b6cb0", "CD202": "#c53030", "EMB48": "#2f855a"}
MARCADORES = {"IRR": "s", "NIR": "^"}
ROTULOS = {"IRR": "Irrigado", "NIR": "Não irrigado"}


def legenda_condicao(ax) -> None:
    handles = [
        plt.Line2D([], [], marker=marcador, linestyle="", color="#4a5568", markersize=7,
                   label=ROTULOS[condicao])
        for condicao, marcador in MARCADORES.items()
    ]
    ax.legend(handles=handles, title="Condição", frameon=False, loc="best")


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top5 = pd.read_csv(ENTRADA / "top5_vip_ipls10_relacoes_cra.csv", sep=";")
    pred = pd.read_csv(ENTRADA / "predicoes_plsr_cra_ipls10.csv", sep=";")
    espectros, _ = base.ler_espectros()
    dados = espectros.merge(ler_cra(), on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    return top5, pred, dados


def plotar_predito_por_dia(pred: pd.DataFrame) -> None:
    for dia, parte in pred.groupby("data_fisiologica", sort=True):
        fig, ax = plt.subplots(figsize=(5.3, 5.0), layout="constrained")
        y, yhat = parte["cra_observado"], parte["cra_predito_loo"]
        for _, linha in parte.iterrows():
            ax.scatter(linha["cra_observado"], linha["cra_predito_loo"], s=62,
                       color=CORES[linha["genotipo"]], marker=MARCADORES[linha["condicao"]],
                       edgecolor="white", linewidth=.7)
        lo, hi = min(y.min(), yhat.min()) - 1, max(y.max(), yhat.max()) + 1
        ax.plot([lo, hi], [lo, hi], "--", color="#4a5568", linewidth=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="CRA observado", ylabel="CRA predito (LOO)",
               title=f"CRA — Real × Predito, {dia}")
        ax.text(.04, .95, f"R² = {r2_score(y, yhat):.3f}\nRMSE = {root_mean_squared_error(y, yhat):.2f}",
                transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0"})
        legenda_condicao(ax)
        fig.savefig(SAIDA / f"{dia.replace('/', '-')}_real_x_predito_loo.png", dpi=220)
        plt.close(fig)


def plotar_relacoes_por_dia(top5: pd.DataFrame, dados: pd.DataFrame) -> None:
    for dia, parte in dados.groupby("data_fisiologica", sort=True):
        fig, eixos = plt.subplots(1, len(top5), figsize=(4.1 * len(top5), 4.5), layout="constrained")
        for ax, (_, banda) in zip(np.atleast_1d(eixos), top5.iterrows()):
            coluna = str(int(banda["banda_nm"]))
            x, y = parte[[coluna]].to_numpy(float), parte["chl_total"].to_numpy(float)
            modelo = LinearRegression().fit(x, y)
            xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            ax.plot(xx, modelo.predict(xx), "--", color="#4a5568", linewidth=1)
            for _, linha in parte.iterrows():
                ax.scatter(linha[coluna], linha["chl_total"], s=62, color=CORES[linha["genotipo"]],
                           marker=MARCADORES[linha["condicao"]], edgecolor="white", linewidth=.7)
            ax.set(xlabel="Reflectância", ylabel="CRA",
                   title=(f"{int(banda['banda_nm'])} nm\nVIP global={banda['vip_plsr_final']:.2f}\n"
                          f"R²={r2_score(y, modelo.predict(x)):.3f}; RMSE={root_mean_squared_error(y, modelo.predict(x)):.2f}"))
        legenda_condicao(np.atleast_1d(eixos)[0])
        fig.suptitle(f"CRA × reflectância por data — bandas VIP globais, {dia}")
        fig.savefig(SAIDA / f"{dia.replace('/', '-')}_relacoes_top5_vip.png", dpi=220)
        plt.close(fig)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    top5, pred, dados = carregar_dados()
    plotar_predito_por_dia(pred)
    plotar_relacoes_por_dia(top5, dados)


if __name__ == "__main__":
    main()
