#!/usr/bin/env python3
"""Repete, para CRA, as quatro estratégias de seleção de bandas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import clorofila_total_spearman_plsr as base
import comparar_selecao_bandas_chltotal as selecao


SAIDA = Path(__file__).resolve().parent / "resultados_cra"


def ler_cra() -> pd.DataFrame:
    bruto = pd.read_excel(base.ARQ_FISIO)
    bruto = bruto.rename(columns=lambda coluna: str(coluna).replace(".", "/"))
    dias_disponiveis = [dia for dia in base.MAPA_DIAS.values() if f"CRA ({dia})" in bruto.columns]
    linhas: list[dict[str, object]] = []
    for _, linha in bruto.iterrows():
        for dia in dias_disponiveis:
            linhas.append(
                {
                    "data_fisiologica": dia,
                    "genotipo": linha["Genótipo"],
                    "condicao": linha["Condição"],
                    "chl_total": pd.to_numeric(linha[f"CRA ({dia})"], errors="coerce"),
                }
            )
    return (
        pd.DataFrame(linhas)
        .dropna(subset=["chl_total"])
        .groupby(["data_fisiologica", "genotipo", "condicao"], as_index=False)
        .agg(chl_total=("chl_total", "mean"), n_repeticoes_fisiologia=("chl_total", "size"))
    )


def plotar_resultados_vip_e_relacoes(
    tabela: pd.DataFrame, dados: pd.DataFrame, top5: pd.DataFrame
) -> None:
    """Salva o perfil Spearman/VIP e as relações CRA × reflectância das top 5."""
    fig, eixos = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, layout="constrained")
    significativa = tabela["p_valor"] < 0.05
    eixos[0].scatter(tabela.loc[~significativa, "banda_nm"], tabela.loc[~significativa, "rho_spearman"],
                     s=5, color="#a0aec0", label="p ≥ 0,05")
    eixos[0].scatter(tabela.loc[significativa, "banda_nm"], tabela.loc[significativa, "rho_spearman"],
                     s=6, color="#c53030", label="p < 0,05")
    eixos[0].axhline(0, color="#4a5568", linewidth=.8)
    eixos[0].set(ylabel="ρ de Spearman", title="CRA × reflectância: correlação por banda (n = 18)")
    eixos[0].legend(frameon=False, ncol=2)

    vip = tabela.dropna(subset=["vip"])
    eixos[1].plot(vip["banda_nm"], vip["vip"], color="#2b6cb0", linewidth=1.1)
    eixos[1].scatter(top5["banda_nm"], top5["vip"], color="#c53030", s=32, zorder=3, label="Top 5 VIP")
    eixos[1].axhline(1, color="#c53030", linestyle="--", linewidth=.9, label="VIP = 1")
    eixos[1].set(xlabel="Comprimento de onda (nm)", ylabel="VIP", title="PLSR: importância das bandas retidas")
    eixos[1].legend(frameon=False, ncol=2)
    fig.savefig(SAIDA / "spearman_vip_cra.png", dpi=220)
    plt.close(fig)

    fig, eixos = plt.subplots(1, len(top5), figsize=(4.1 * len(top5), 4.4), layout="constrained")
    eixos = np.atleast_1d(eixos)
    cores = plt.get_cmap("tab10")
    for eixo, (_, banda) in zip(eixos, top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        for indice, dia in enumerate(sorted(dados["data_fisiologica"].unique())):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(dados.loc[parte, coluna], dados.loc[parte, "chl_total"],
                         s=42, color=cores(indice), label=dia, edgecolor="white", linewidth=.5)
        x = dados[coluna].to_numpy(float)
        y = dados["chl_total"].to_numpy(float)
        inclinacao, intercepto = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        eixo.plot(xx, inclinacao * xx + intercepto, "--", color="#4a5568", linewidth=1)
        eixo.set(
            xlabel="Reflectância", ylabel="CRA",
            title=(f"#{int(banda['rank_vip'])}: {coluna} nm\n"
                   f"VIP={banda['vip']:.2f}; ρ={banda['rho_spearman']:.2f}; p={banda['p_valor']:.3f}"),
        )
    eixos[0].legend(title="Data", frameon=False, fontsize=7, title_fontsize=8)
    fig.suptitle("CRA × reflectância nas cinco bandas com maior VIP", fontsize=13)
    fig.savefig(SAIDA / "relacao_cra_top5_vip.png", dpi=220)
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    # As funções compartilhadas usam este destino para o gráfico.
    selecao.SAIDA = SAIDA
    fisio = ler_cra()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    dados.to_csv(SAIDA / "dados_agregados_cra.csv", index=False, sep=";")
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({
        "banda_nm": np.array(bandas, int),
        "rho_spearman": [r[0] for r in correlacoes],
        "p_valor": [r[1] for r in correlacoes],
    })
    retidas = tabela.loc[tabela["p_valor"] < .05].copy()
    xr = dados[retidas["banda_nm"].astype(str)].to_numpy(float)
    componentes, rmse_loo = base.melhor_n_componentes(xr, y)
    modelo = PLSRegression(n_components=componentes, scale=True).fit(xr, y)
    retidas["vip"] = base.calcular_vip(modelo)
    tabela["vip"] = np.nan
    tabela.loc[retidas.index, "vip"] = retidas["vip"]
    retidas.to_csv(SAIDA / "cra_spearman_vip.csv", index=False, sep=";")
    top5 = retidas.nlargest(5, "vip").copy()
    top5.insert(0, "rank_vip", range(1, len(top5) + 1))
    top5["direcao_relacao"] = np.where(top5["rho_spearman"] > 0, "positiva", "negativa")
    top5.to_csv(SAIDA / "top5_vip_relacao_cra.csv", index=False, sep=";")
    plotar_resultados_vip_e_relacoes(tabela, dados, top5)

    pred_ajuste = modelo.predict(xr).ravel()
    pred_loo = cross_val_predict(
        PLSRegression(n_components=componentes, scale=True), xr, y, cv=LeaveOneOut()
    ).ravel()
    predicoes = dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].rename(
        columns={"chl_total": "cra_observado"}
    )
    predicoes["cra_predito_ajuste"] = pred_ajuste
    predicoes["cra_predito_loo"] = pred_loo
    predicoes.to_csv(SAIDA / "predicoes_plsr_cra.csv", index=False, sep=";")
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    cores = plt.get_cmap("tab10")
    for eixo, previsto, titulo in (
        (eixos[0], pred_ajuste, "Ajuste"),
        (eixos[1], pred_loo, "Validação leave-one-out"),
    ):
        for indice, dia in enumerate(sorted(dados["data_fisiologica"].unique())):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(y[parte], previsto[parte], color=cores(indice), s=44, label=dia)
        limite_min = min(y.min(), previsto.min()) - 1
        limite_max = max(y.max(), previsto.max()) + 1
        eixo.plot([limite_min, limite_max], [limite_min, limite_max], "--", color="#4a5568", linewidth=1)
        eixo.set(xlim=(limite_min, limite_max), ylim=(limite_min, limite_max), xlabel="CRA observado", ylabel="CRA predito", title=titulo)
        eixo.text(
            .04, .94,
            f"R² = {r2_score(y, previsto):.3f}\nRMSE = {root_mean_squared_error(y, previsto):.2f}",
            transform=eixo.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.3"},
        )
    eixos[0].legend(title="Dia", frameon=False, fontsize=8, title_fontsize=9)
    fig.suptitle(f"PLSR para CRA — {len(retidas)} bandas Spearman (p < 0,05); {componentes} componentes")
    fig.savefig(SAIDA / "regressao_plsr_cra_observado_vs_predito.png", dpi=220)
    plt.close(fig)

    selecoes = [
        selecao.adicionar_metodo("janela_minima_20_nm", selecao.janela_minima(retidas)),
        selecao.representantes_colinearidade(retidas, dados),
        selecao.interval_pls(retidas, dados),
        selecao.picos_locais(retidas),
    ]
    resultado = pd.concat(selecoes, ignore_index=True)
    resultado.to_csv(SAIDA / "comparacao_quatro_metodos_bandas_selecionadas_cra.csv", index=False, sep=";")
    selecao.plotar(retidas, resultado, parametro="CRA")
    pd.DataFrame([{
        "n_grupos_dia_genotipo_condicao": len(dados),
        "n_bandas_testadas": len(bandas),
        "n_bandas_spearman_p_lt_0_05": len(retidas),
        "componentes_plsr": componentes,
        "r2_ajuste_plsr": r2_score(y, pred_ajuste),
        "rmse_ajuste_plsr": root_mean_squared_error(y, pred_ajuste),
        "r2_loo_plsr": r2_score(y, pred_loo),
        "rmse_loo_plsr": root_mean_squared_error(y, pred_loo),
    }]).to_csv(SAIDA / "resumo_cra.csv", index=False, sep=";")


if __name__ == "__main__":
    main()
