#!/usr/bin/env python3
"""CRA: Spearman, iPLS por intervalos de 20 nm e PLSR/VIP final.

O Spearman (p < 0,05) é reportado como diagnóstico de associação. O iPLS
avalia todas as janelas de 20 nm por validação leave-one-out e retém a banda de
maior VIP em cada janela. A escolha final é gulosa por menor RMSE do intervalo,
com |r de Pearson| < 0,80 entre cada nova banda e as já escolhidas. Esse passo
é necessário porque o filtro Spearman isolado retém apenas três regiões
contíguas e não permite formar cinco bandas pouco colineares.
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
import comparar_selecao_bandas_chltotal as selecao
from comparar_selecao_bandas_cra import ler_cra


SAIDA = Path(__file__).resolve().parent / "resultados_cra_interval_pls"
LIMIAR_ABS_CORRELACAO = 0.80


def selecionar_ipls_diverso(
    dados: pd.DataFrame, bandas: list[str], y: np.ndarray, tabela: pd.DataFrame,
    largura_intervalo_nm: int = selecao.LARGURA_JANELA_NM,
    limiar_correlacao: float | None = LIMIAR_ABS_CORRELACAO,
    n_bandas: int = 5,
) -> pd.DataFrame:
    """Representante VIP por janela iPLS, seguido de filtro de colinearidade."""
    registros: list[dict[str, float | int | str]] = []
    for inicio in range(350, 2501, largura_intervalo_nm):
        fim = min(inicio + largura_intervalo_nm - 1, 2500)
        janela = [str(banda) for banda in range(inicio, fim + 1) if str(banda) in bandas]
        if not janela:
            continue
        x_janela = dados[janela].to_numpy(float)
        n_componentes, rmse = base.melhor_n_componentes(x_janela, y)
        vip_janela = base.calcular_vip(PLSRegression(n_components=n_componentes, scale=True).fit(x_janela, y))
        indice = int(np.argmax(vip_janela))
        banda = int(janela[indice])
        spearman = tabela.loc[tabela["banda_nm"].eq(banda)].iloc[0]
        registros.append({
            "banda_nm": banda, "rho_spearman": spearman["rho_spearman"], "p_valor": spearman["p_valor"],
            "vip_intervalo": vip_janela[indice], "rmse_loo_intervalo": rmse,
            "intervalo_nm": f"{inicio}-{fim}", "componentes_intervalo": n_componentes,
        })
    candidatas = pd.DataFrame(registros).sort_values("rmse_loo_intervalo")
    escolhidas: list[pd.Series] = []
    for _, candidata in candidatas.iterrows():
        coluna = str(int(candidata["banda_nm"]))
        if limiar_correlacao is None or all(
            abs(dados[[coluna, str(int(escolhida["banda_nm"]))]].corr().iloc[0, 1]) < limiar_correlacao
            for escolhida in escolhidas
        ):
            escolhidas.append(candidata)
        if len(escolhidas) == n_bandas:
            break
    if len(escolhidas) != n_bandas:
        raise ValueError(f"Não foi possível obter {n_bandas} bandas iPLS sob o limiar de colinearidade definido.")
    return pd.DataFrame(escolhidas)


def plotar_predicoes(y: np.ndarray, ajuste: np.ndarray, loo: np.ndarray, dados: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    cores = plt.get_cmap("tab10")
    for eixo, previsto, titulo in ((eixos[0], ajuste, "Ajuste"), (eixos[1], loo, "Validação leave-one-out")):
        for indice, dia in enumerate(sorted(dados["data_fisiologica"].unique())):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(y[parte], previsto[parte], color=cores(indice), s=44, label=dia)
        minimo, maximo = min(y.min(), previsto.min()) - 1, max(y.max(), previsto.max()) + 1
        eixo.plot([minimo, maximo], [minimo, maximo], "--", color="#4a5568", linewidth=1)
        eixo.set(xlim=(minimo, maximo), ylim=(minimo, maximo), xlabel="CRA observado", ylabel="CRA predito", title=titulo)
        eixo.text(.04, .94, f"R² = {r2_score(y, previsto):.3f}\nRMSE = {root_mean_squared_error(y, previsto):.2f}",
                  transform=eixo.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.3"})
    eixos[0].legend(title="Data", frameon=False, fontsize=8, title_fontsize=9)
    fig.suptitle("PLSR para CRA — cinco bandas selecionadas por iPLS")
    fig.savefig(SAIDA / "regressao_plsr_ipls_cra_observado_vs_predito.png", dpi=220)
    plt.close(fig)


def plotar_selecao(tabela: pd.DataFrame, top5: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, layout="constrained")
    significativa = tabela["p_valor"] < .05
    eixos[0].scatter(tabela.loc[~significativa, "banda_nm"], tabela.loc[~significativa, "rho_spearman"], s=5, color="#a0aec0", label="p ≥ 0,05")
    eixos[0].scatter(tabela.loc[significativa, "banda_nm"], tabela.loc[significativa, "rho_spearman"], s=6, color="#c53030", label="p < 0,05")
    eixos[0].axhline(0, color="#4a5568", linewidth=.8)
    eixos[0].set(ylabel="ρ de Spearman", title="CRA × reflectância: filtro inicial (n = 18)")
    eixos[0].legend(frameon=False, ncol=2)
    eixos[1].scatter(top5["banda_nm"], top5["vip_plsr_final"], s=55, color="#c53030", zorder=3, label="Top 5 VIP final")
    for _, linha in top5.iterrows():
        eixos[1].annotate(f"{int(linha['banda_nm'])} nm\n{linha['intervalo_nm']}", (linha["banda_nm"], linha["vip_plsr_final"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    eixos[1].axhline(1, color="#2b6cb0", linestyle="--", linewidth=.9, label="VIP = 1")
    eixos[1].set(xlabel="Comprimento de onda (nm)", ylabel="VIP final", title="PLSR final: bandas de intervalos iPLS distintos")
    eixos[1].legend(frameon=False)
    fig.savefig(SAIDA / "spearman_ipls_vip_cra.png", dpi=220)
    plt.close(fig)


def plotar_relacoes(dados: pd.DataFrame, top5: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(1, len(top5), figsize=(4.1 * len(top5), 4.4), layout="constrained")
    eixos = np.atleast_1d(eixos)
    cores = plt.get_cmap("tab10")
    for eixo, (_, banda) in zip(eixos, top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        for indice, dia in enumerate(sorted(dados["data_fisiologica"].unique())):
            parte = dados["data_fisiologica"].eq(dia)
            eixo.scatter(dados.loc[parte, coluna], dados.loc[parte, "chl_total"], s=42, color=cores(indice), label=dia, edgecolor="white", linewidth=.5)
        x, y = dados[coluna].to_numpy(float), dados["chl_total"].to_numpy(float)
        inclinacao, intercepto = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        eixo.plot(xx, inclinacao * xx + intercepto, "--", color="#4a5568", linewidth=1)
        eixo.set(xlabel="Reflectância", ylabel="CRA", title=(f"#{int(banda['rank_vip_final'])}: {coluna} nm\nVIP={banda['vip_plsr_final']:.2f}; ρ={banda['rho_spearman']:.2f}; p={banda['p_valor']:.3f}"))
    eixos[0].legend(title="Data", frameon=False, fontsize=7, title_fontsize=8)
    fig.suptitle("CRA × reflectância: top 5 VIP após iPLS", fontsize=13)
    fig.savefig(SAIDA / "relacao_cra_top5_vip_interval_pls.png", dpi=220)
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    fisiologia = ler_cra()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisiologia, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.array(bandas, dtype=int), "rho_spearman": [v[0] for v in correlacoes], "p_valor": [v[1] for v in correlacoes]})
    retidas = tabela.loc[tabela["p_valor"] < .05].copy()
    escolhidas = selecionar_ipls_diverso(dados, bandas, y, tabela)
    bandas_finais = escolhidas["banda_nm"].astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)
    n_final, _ = base.melhor_n_componentes(x_final, y)
    modelo = PLSRegression(n_components=n_final, scale=True).fit(x_final, y)
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
    top5 = escolhidas.sort_values("vip_plsr_final", ascending=False).reset_index(drop=True)
    top5.insert(0, "rank_vip_final", range(1, len(top5) + 1))
    top5["direcao_relacao"] = np.where(top5["rho_spearman"] > 0, "positiva", "negativa")
    top5.to_csv(SAIDA / "top5_vip_interval_pls_cra.csv", sep=";", index=False)
    tabela.to_csv(SAIDA / "spearman_cra.csv", sep=";", index=False)

    ajuste = modelo.predict(x_final).ravel()
    loo = cross_val_predict(PLSRegression(n_components=n_final, scale=True), x_final, y, cv=LeaveOneOut()).ravel()
    predicoes = dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].rename(columns={"chl_total": "cra_observado"})
    predicoes["cra_predito_ajuste"] = ajuste
    predicoes["cra_predito_loo"] = loo
    predicoes.to_csv(SAIDA / "predicoes_plsr_interval_pls_cra.csv", sep=";", index=False)

    matriz_correlacao = dados[[str(banda) for banda in bandas_finais]].corr(method="pearson")
    matriz_correlacao.index = [f"{banda} nm" for banda in bandas_finais]
    matriz_correlacao.columns = matriz_correlacao.index
    matriz_correlacao.to_csv(SAIDA / "correlacao_pearson_top5_interval_pls_cra.csv", sep=";")
    valores_fora_diagonal = matriz_correlacao.to_numpy()[np.triu_indices(len(bandas_finais), k=1)]
    pd.DataFrame([{
        "n_grupos": len(dados), "n_bandas_testadas": len(bandas), "n_bandas_spearman_p_lt_0_05": len(retidas),
        "largura_intervalo_nm": selecao.LARGURA_JANELA_NM, "limiar_abs_correlacao_pearson": LIMIAR_ABS_CORRELACAO,
        "n_bandas_finais": len(bandas_finais), "componentes_plsr_final": n_final,
        "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": root_mean_squared_error(y, ajuste),
        "r2_loo": r2_score(y, loo), "rmse_loo": root_mean_squared_error(y, loo),
        "max_abs_correlacao_pearson_entre_bandas": np.abs(valores_fora_diagonal).max(),
        "media_abs_correlacao_pearson_entre_bandas": np.abs(valores_fora_diagonal).mean(),
    }]).to_csv(SAIDA / "resumo_interval_pls_cra.csv", sep=";", index=False)
    plotar_predicoes(y, ajuste, loo, dados)
    plotar_selecao(tabela, top5)
    plotar_relacoes(dados, top5)


if __name__ == "__main__":
    main()
