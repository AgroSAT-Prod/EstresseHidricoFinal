#!/usr/bin/env python3
"""CHL Total: Spearman p < 0,001, iPLS de 10 nm, PLSR e VIP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import clorofila_total_spearman_plsr as base
import cra_interval_pls_vip as ipls


SAIDA = Path(__file__).resolve().parent / "resultados_chltotal_spearman_p0001_ipls10"
LARGURA_JANELA_NM = 10
P_LIMIAR = 0.001


def plotar_predicoes(y: np.ndarray, ajuste: np.ndarray, loo: np.ndarray, dados: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    cores = plt.get_cmap("tab10")
    for eixo, previsto, titulo in ((eixos[0], ajuste, "Ajuste"), (eixos[1], loo, "Validação leave-one-out")):
        for indice, dia in enumerate(base.MAPA_DIAS.values()):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(y[parte], previsto[parte], color=cores(indice), s=32, label=dia)
        minimo, maximo = min(y.min(), previsto.min()) - .5, max(y.max(), previsto.max()) + .5
        eixo.plot([minimo, maximo], [minimo, maximo], "--", color="#4a5568", linewidth=1)
        eixo.set(xlim=(minimo, maximo), ylim=(minimo, maximo), xlabel="CHL Total observado", ylabel="CHL Total predito", title=titulo)
        eixo.text(.04, .94, f"R² = {r2_score(y, previsto):.3f}\nRMSE = {root_mean_squared_error(y, previsto):.2f}", transform=eixo.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.3"})
    eixos[0].legend(title="Data", frameon=False, fontsize=7, title_fontsize=8)
    fig.suptitle("PLSR para CHL Total — cinco bandas selecionadas por iPLS (10 nm)")
    fig.savefig(SAIDA / "regressao_plsr_chltotal_ipls10_observado_vs_predito.png", dpi=220)
    plt.close(fig)


def plotar_relacoes(dados: pd.DataFrame, top5: pd.DataFrame) -> pd.DataFrame:
    registros = []
    fig, eixos = plt.subplots(1, len(top5), figsize=(4.15 * len(top5), 4.6), layout="constrained")
    eixos = np.atleast_1d(eixos)
    cores = plt.get_cmap("tab10")
    for eixo, (_, banda) in zip(eixos, top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        x, y = dados[[coluna]].to_numpy(float), dados["chl_total"].to_numpy(float)
        regressao = LinearRegression().fit(x, y)
        previsto = regressao.predict(x)
        r2, rmse = r2_score(y, previsto), root_mean_squared_error(y, previsto)
        registros.append({"rank_vip": int(banda["rank_vip"]), "banda_nm": int(banda["banda_nm"]), "r2_regressao_linear": r2, "rmse_regressao_linear": rmse})
        for indice, dia in enumerate(base.MAPA_DIAS.values()):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(dados.loc[parte, "chl_total"], dados.loc[parte, coluna], s=25, color=cores(indice), label=dia)
        xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        # Métricas da predição de CHL Total; a curva é exibida com os eixos invertidos.
        eixo.plot(regressao.predict(xx), xx, "--", color="#4a5568", linewidth=1)
        eixo.set(xlabel="CHL Total", ylabel="Reflectância", title=(f"#{int(banda['rank_vip'])}: {coluna} nm\nVIP={banda['vip_plsr_final']:.2f}; ρ={banda['rho_spearman']:.2f}; p={banda['p_valor']:.3g}\nR²={r2:.3f}; RMSE={rmse:.2f}"))
    eixos[0].legend(title="Data", frameon=False, fontsize=6, title_fontsize=7)
    fig.suptitle("CHL Total × reflectância: top 5 VIP após Spearman + iPLS (10 nm)", fontsize=13)
    fig.savefig(SAIDA / "relacao_reflectancia_chltotal_top5_vip_ipls10.png", dpi=220)
    plt.close(fig)
    return pd.DataFrame(registros)


def plotar_spearman_vip(tabela: pd.DataFrame, top5: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, layout="constrained")
    significativa = tabela["p_valor"] < P_LIMIAR
    eixos[0].scatter(tabela.loc[~significativa, "banda_nm"], tabela.loc[~significativa, "rho_spearman"], s=4, color="#a0aec0", label="p ≥ 0,001")
    eixos[0].scatter(tabela.loc[significativa, "banda_nm"], tabela.loc[significativa, "rho_spearman"], s=5, color="#c53030", label="p < 0,001")
    eixos[0].axhline(0, color="#4a5568", linewidth=.8)
    eixos[0].set(ylabel="ρ de Spearman", title="CHL Total × reflectância: filtro Spearman (n = 42)")
    eixos[0].legend(frameon=False, ncol=2)
    eixos[1].scatter(top5["banda_nm"], top5["vip_plsr_final"], color="#c53030", s=55, label="Top 5 VIP final")
    eixos[1].axhline(1, color="#2b6cb0", linestyle="--", linewidth=.9, label="VIP = 1")
    for _, linha in top5.iterrows():
        eixos[1].annotate(f"{int(linha['banda_nm'])} nm", (linha["banda_nm"], linha["vip_plsr_final"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    eixos[1].set(xlabel="Comprimento de onda (nm)", ylabel="VIP final", title="PLSR final após iPLS (janelas de 10 nm)")
    eixos[1].legend(frameon=False)
    fig.savefig(SAIDA / "spearman_ipls10_vip_chltotal.png", dpi=220)
    plt.close(fig)


def main() -> None:
    global P_LIMIAR, SAIDA
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-limiar", type=float, default=P_LIMIAR)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    if not 0 < args.p_limiar < 1:
        raise ValueError("--p-limiar deve estar entre 0 e 1.")
    P_LIMIAR, SAIDA = args.p_limiar, args.saida
    SAIDA.mkdir(exist_ok=True)
    fisio = base.ler_fisiologia()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.array(bandas, dtype=int), "rho_spearman": [r[0] for r in correlacoes], "p_valor": [r[1] for r in correlacoes]})
    retidas = tabela.loc[tabela["p_valor"] < P_LIMIAR]
    escolhidas = ipls.selecionar_ipls_diverso(dados, retidas["banda_nm"].astype(str).tolist(), y, tabela, largura_intervalo_nm=LARGURA_JANELA_NM, limiar_correlacao=None)
    bandas_finais = escolhidas["banda_nm"].astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)
    componentes, _ = base.melhor_n_componentes(x_final, y)
    modelo = PLSRegression(n_components=componentes, scale=True).fit(x_final, y)
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
    top5 = escolhidas.sort_values("vip_plsr_final", ascending=False).reset_index(drop=True)
    top5.insert(0, "rank_vip", range(1, len(top5) + 1))
    top5["direcao_relacao"] = np.where(top5["rho_spearman"] > 0, "positiva", "negativa")

    ajuste = modelo.predict(x_final).ravel()
    loo = cross_val_predict(PLSRegression(n_components=componentes, scale=True), x_final, y, cv=LeaveOneOut()).ravel()
    predicoes = dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].rename(columns={"chl_total": "chltotal_observado"})
    predicoes["chltotal_predito_ajuste"] = ajuste
    predicoes["chltotal_predito_loo"] = loo
    predicoes.to_csv(SAIDA / "predicoes_plsr_chltotal_ipls10.csv", sep=";", index=False)
    relacoes = plotar_relacoes(dados, top5)
    top5 = top5.merge(relacoes, on=["rank_vip", "banda_nm"], validate="one_to_one")
    top5.to_csv(SAIDA / "top5_vip_ipls10_relacoes_chltotal.csv", sep=";", index=False)
    tabela.to_csv(SAIDA / "spearman_chltotal.csv", sep=";", index=False)
    plotar_predicoes(y, ajuste, loo, dados)
    plotar_spearman_vip(tabela, top5)
    sufixo_limiar = f"{P_LIMIAR:.6f}".rstrip("0").rstrip(".").replace(".", "_")
    pd.DataFrame([{
        "n_grupos": len(dados), "n_bandas_testadas": len(bandas), f"n_bandas_spearman_p_lt_{sufixo_limiar}": len(retidas),
        "largura_janela_ipls_nm": LARGURA_JANELA_NM, "n_bandas_finais": len(top5), "componentes_plsr": componentes,
        "r2_ajuste_plsr": r2_score(y, ajuste), "rmse_ajuste_plsr": root_mean_squared_error(y, ajuste),
        "r2_loo_plsr": r2_score(y, loo), "rmse_loo_plsr": root_mean_squared_error(y, loo),
    }]).to_csv(SAIDA / "resumo_plsr_chltotal_ipls10.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
