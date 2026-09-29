#!/usr/bin/env python3
"""Relação entre CRA e reflectância pré-processada + normalizada (SNV).

Usa o estágio final da pipeline: correção de emendas, recorte 400–2450 nm,
suavização Savitzky–Golay (11, ordem 2) e SNV. A unidade é a média de
genótipo × condição em cada data com CRA disponível, no turno da manhã.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import clorofila_total_spearman_plsr as base
from comparar_selecao_bandas_cra import ler_cra


ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "dataset" / "Unificada13052026_normalizado.csv"
SAIDA = Path(__file__).resolve().parent / "resultados_cra_normalizado"


def ler_espectros_snv() -> tuple[pd.DataFrame, list[str]]:
    dados = pd.read_csv(ENTRADA, sep=";")
    bandas = [str(i) for i in range(400, 2451) if str(i) in dados.columns]
    dados = dados.loc[dados["data_coleta"].isin(base.MAPA_DIAS) & dados["turno"].eq("manha")].copy()
    dados["data_fisiologica"] = dados["data_coleta"].map(base.MAPA_DIAS)
    dados["condicao"] = dados["condicao"].map({"IRRIG": "IRR", "NIRRIG": "NIR"})
    agrupado = dados.groupby(["data_fisiologica", "genotipo", "condicao"], as_index=False)[bandas].mean()
    return agrupado, bandas


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    espectros, bandas = ler_espectros_snv()
    dados = espectros.merge(ler_cra(), on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)  # coluna interna retornada por ler_cra
    x = dados[bandas].to_numpy(float)
    resultados = [base.spearman_com_p_exato(x[:, indice], y) for indice in range(x.shape[1])]
    tabela = pd.DataFrame({
        "banda_nm": np.array(bandas, dtype=int),
        "rho_spearman": [item[0] for item in resultados],
        "p_valor": [item[1] for item in resultados],
    })
    tabela["significativa"] = tabela["p_valor"] < .05
    selecionadas = tabela.loc[tabela["significativa"], "banda_nm"].astype(str).tolist()
    xs = dados[selecionadas].to_numpy(float)
    componentes, rmse_loo = base.melhor_n_componentes(xs, y)
    modelo = PLSRegression(n_components=componentes, scale=True).fit(xs, y)
    pred_ajuste = modelo.predict(xs).ravel()
    pred_loo = cross_val_predict(PLSRegression(n_components=componentes, scale=True), xs, y, cv=LeaveOneOut()).ravel()
    tabela["vip"] = np.nan
    tabela.loc[tabela["significativa"], "vip"] = base.calcular_vip(modelo)
    tabela.to_csv(SAIDA / "cra_snv_spearman_vip.csv", sep=";", index=False)
    dados_saida = dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].rename(columns={"chl_total": "cra_observado"})
    dados_saida["cra_predito_ajuste"] = pred_ajuste
    dados_saida["cra_predito_loo"] = pred_loo
    dados_saida.to_csv(SAIDA / "predicoes_plsr_cra_snv.csv", sep=";", index=False)
    pd.DataFrame([{
        "n_grupos": len(dados), "n_bandas_snv": len(bandas), "n_bandas_spearman_p_lt_0_05": len(selecionadas),
        "componentes_plsr": componentes, "r2_ajuste": r2_score(y, pred_ajuste),
        "rmse_ajuste": root_mean_squared_error(y, pred_ajuste), "r2_loo": r2_score(y, pred_loo),
        "rmse_loo": root_mean_squared_error(y, pred_loo),
    }]).to_csv(SAIDA / "resumo_plsr_cra_snv.csv", sep=";", index=False)

    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    # Curvas já normalizadas: cor codifica CRA observado.
    ordem = np.argsort(y)
    linhas = eixos[0].plot(np.array(bandas, int), x[ordem].T, linewidth=.8, alpha=.85)
    mapa = plt.get_cmap("viridis")
    normalizador = plt.Normalize(y.min(), y.max())
    for linha, cra in zip(linhas, y[ordem]):
        linha.set_color(mapa(normalizador(cra)))
    fig.colorbar(plt.cm.ScalarMappable(norm=normalizador, cmap=mapa), ax=eixos[0], label="CRA observado")
    eixos[0].set(title="Curvas processadas e normalizadas (SNV)", xlabel="Comprimento de onda (nm)", ylabel="Reflectância SNV")

    for indice, dia in enumerate(["24/02", "02/03", "03/03"]):
        parte = dados["data_fisiologica"].eq(dia)
        eixos[1].scatter(y[parte], pred_loo[parte], s=45, label=dia)
    minimo, maximo = min(y.min(), pred_loo.min()) - 1, max(y.max(), pred_loo.max()) + 1
    eixos[1].plot([minimo, maximo], [minimo, maximo], "--", color="#4a5568", linewidth=1)
    eixos[1].set(xlim=(minimo, maximo), ylim=(minimo, maximo), title="PLSR: CRA observado × predito (LOO)", xlabel="CRA observado", ylabel="CRA predito")
    eixos[1].legend(title="Dia", frameon=False)
    eixos[1].text(.04, .94, f"R² LOO = {r2_score(y, pred_loo):.3f}\nRMSE LOO = {root_mean_squared_error(y, pred_loo):.2f}", transform=eixos[1].transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.3"})
    fig.suptitle(f"CRA × reflectância SNV — {len(selecionadas)} bandas Spearman (p < 0,05); {componentes} componentes PLSR")
    fig.savefig(SAIDA / "relacao_cra_reflectancia_snv.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
