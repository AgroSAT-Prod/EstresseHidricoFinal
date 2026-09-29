#!/usr/bin/env python3
"""CRA: Spearman p < 0,05, iPLS de 10 nm e filtro de redundância."""

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
from comparar_selecao_bandas_cra import ler_cra
import cra_interval_pls_vip as graficos


SAIDA = Path(__file__).resolve().parent / "resultados_cra_spearman_ipls10_sem_redundancia_r080"
LARGURA_JANELA_NM = 10
LIMIAR_ABS_CORRELACAO = 0.80


def plotar_relacoes_com_metricas(dados: pd.DataFrame, top5: pd.DataFrame) -> pd.DataFrame:
    """Relações univariadas CRA × reflectância, com R² e RMSE de ajuste."""
    registros = []
    fig, eixos = plt.subplots(1, len(top5), figsize=(4.15 * len(top5), 4.6), layout="constrained")
    eixos = np.atleast_1d(eixos)
    cores = plt.get_cmap("tab10")
    for eixo, (_, banda) in zip(eixos, top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        x = dados[[coluna]].to_numpy(float)
        y = dados["chl_total"].to_numpy(float)
        regressao = LinearRegression().fit(x, y)
        previsto = regressao.predict(x)
        r2, rmse = r2_score(y, previsto), root_mean_squared_error(y, previsto)
        registros.append({"rank_vip": int(banda["rank_vip"]), "banda_nm": int(banda["banda_nm"]), "r2_regressao_linear": r2, "rmse_regressao_linear": rmse})
        for indice, dia in enumerate(sorted(dados["data_fisiologica"].unique())):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(dados.loc[parte, coluna], dados.loc[parte, "chl_total"], s=42, color=cores(indice), label=dia, edgecolor="white", linewidth=.5)
        xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        eixo.plot(xx, regressao.predict(xx), "--", color="#4a5568", linewidth=1)
        eixo.set(xlabel="Reflectância", ylabel="CRA", title=(f"#{int(banda['rank_vip'])}: {coluna} nm\nVIP={banda['vip_plsr_final']:.2f}; ρ={banda['rho_spearman']:.2f}; p={banda['p_valor']:.3f}\nR²={r2:.3f}; RMSE={rmse:.2f}"))
    eixos[0].legend(title="Data", frameon=False, fontsize=7, title_fontsize=8)
    fig.suptitle("CRA × reflectância: top 5 VIP após Spearman + iPLS (10 nm)", fontsize=13)
    fig.savefig(SAIDA / "relacao_reflectancia_cra_top5_vip_ipls10.png", dpi=220)
    plt.close(fig)
    return pd.DataFrame(registros)


def main() -> None:
    global SAIDA
    parser = argparse.ArgumentParser()
    parser.add_argument("--limiar-correlacao", type=float, default=LIMIAR_ABS_CORRELACAO)
    parser.add_argument("--n-bandas", type=int, default=5)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    if not 0 < args.limiar_correlacao < 1:
        raise ValueError("--limiar-correlacao deve estar entre 0 e 1.")
    if args.n_bandas < 1:
        raise ValueError("--n-bandas deve ser maior que zero.")
    SAIDA = args.saida
    SAIDA.mkdir(exist_ok=True)
    # Reutiliza os gráficos de predição e de seleção, gravando neste resultado.
    graficos.SAIDA = SAIDA
    fisio = ler_cra()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    resultados = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.array(bandas, dtype=int), "rho_spearman": [r[0] for r in resultados], "p_valor": [r[1] for r in resultados]})
    retidas = tabela.loc[tabela["p_valor"] < .05].copy()

    # Representantes iPLS são ordenados pelo RMSE LOO do intervalo. Cada nova
    # banda só entra se |r de Pearson| for menor que o limiar com as já aceitas.
    bandas_retidas = retidas["banda_nm"].astype(str).tolist()
    escolhidas = graficos.selecionar_ipls_diverso(
        dados, bandas_retidas, y, tabela, largura_intervalo_nm=LARGURA_JANELA_NM,
        limiar_correlacao=args.limiar_correlacao,
        n_bandas=args.n_bandas,
    )
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
    predicoes = dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].rename(columns={"chl_total": "cra_observado"})
    predicoes["cra_predito_ajuste"] = ajuste
    predicoes["cra_predito_loo"] = loo
    predicoes.to_csv(SAIDA / "predicoes_plsr_cra_ipls10.csv", sep=";", index=False)
    relacoes = plotar_relacoes_com_metricas(dados, top5)
    top5 = top5.merge(relacoes, on=["rank_vip", "banda_nm"], validate="one_to_one")
    top5.to_csv(SAIDA / "top5_vip_ipls10_relacoes_cra.csv", sep=";", index=False)
    tabela.to_csv(SAIDA / "spearman_cra.csv", sep=";", index=False)
    graficos.plotar_predicoes(y, ajuste, loo, dados)
    graficos.plotar_selecao(tabela, top5.rename(columns={"rank_vip": "rank_vip_final"}))

    matriz_correlacao = dados[[str(banda) for banda in bandas_finais]].corr(method="pearson")
    matriz_correlacao.index = [f"{banda} nm" for banda in bandas_finais]
    matriz_correlacao.columns = matriz_correlacao.index
    matriz_correlacao.to_csv(SAIDA / "correlacao_pearson_bandas_finais.csv", sep=";")
    valores_fora_diagonal = matriz_correlacao.to_numpy()[np.triu_indices(len(bandas_finais), k=1)]
    pd.DataFrame([{
        "n_grupos": len(dados), "n_bandas_testadas": len(bandas), "n_bandas_spearman_p_lt_0_05": len(retidas),
        "largura_janela_ipls_nm": LARGURA_JANELA_NM, "limiar_abs_correlacao_pearson": args.limiar_correlacao,
        "max_abs_correlacao_pearson_entre_bandas": np.abs(valores_fora_diagonal).max(),
        "media_abs_correlacao_pearson_entre_bandas": np.abs(valores_fora_diagonal).mean(),
        "n_bandas_finais": len(top5), "componentes_plsr": componentes,
        "r2_ajuste_plsr": r2_score(y, ajuste), "rmse_ajuste_plsr": root_mean_squared_error(y, ajuste),
        "r2_loo_plsr": r2_score(y, loo), "rmse_loo_plsr": root_mean_squared_error(y, loo),
    }]).to_csv(SAIDA / "resumo_plsr_cra_ipls10.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
