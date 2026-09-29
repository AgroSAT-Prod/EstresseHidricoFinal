#!/usr/bin/env python3
"""Associa curvas espectrais matinais à clorofila total por dia.

A planilha fisiológica não possui planta, bloco ou outra chave que permita
parear cada uma das curvas individuais. Por isso, a unidade experimental desta
análise é a média de cada combinação genótipo × condição (n=6 por dia).

Etapas para cada data:
1. Spearman entre CHL TOTAL e cada comprimento de onda; retenção p < 0,05.
2. PLSR usando somente as bandas retidas; número de componentes definido por
   menor RMSE em validação leave-one-out; cálculo de VIP.

Os resultados são exploratórios: as seis médias não são suficientes para
validar um modelo preditivo ou para corrigir adequadamente os múltiplos testes.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
ARQ_ESPECTRAL = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
ARQ_FISIO = ROOT / "dataset" / "FisiologicosEduarda.xlsx"
SAIDA = Path(__file__).resolve().parent / "resultados_clorofila_total"

# Confirmado para D02, D06 e D09 por análises existentes; os demais dias
# preenchem a sequência de coletas que existe na base espectral.
MAPA_DIAS = {
    "D02M": "23/02",
    "D03M": "24/02",
    "D04M": "25/02",
    "D05M": "26/02",
    "D06M": "27/02",
    "D09M": "02/03",
    "D10M": "03/03",
}

# Com n=6, a aproximação t usada usualmente por spearmanr é pouco adequada.
# A distribuição nula exata tem somente 6! = 720 permutações.
_RANKS = np.arange(1, 7, dtype=float)
_RHO_NULO_EXATO = np.array(
    [np.corrcoef(_RANKS, permutacao)[0, 1] for permutacao in permutations(_RANKS)]
)


def spearman_com_p_exato(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    resultado = spearmanr(x, y)
    rho = float(resultado.statistic)
    if np.isnan(rho):
        return rho, np.nan
    if len(x) != 6:
        return rho, float(resultado.pvalue)
    # Não há empates nas médias usadas aqui; a distribuição é exata sob H0.
    p = float(np.mean(np.abs(_RHO_NULO_EXATO) >= abs(rho) - 1e-12))
    return rho, p


def ler_fisiologia() -> pd.DataFrame:
    bruto = pd.read_excel(ARQ_FISIO)
    # A própria planilha alterna ponto e barra em alguns cabeçalhos de data.
    bruto = bruto.rename(columns=lambda coluna: str(coluna).replace(".", "/"))
    linhas: list[dict[str, object]] = []
    for _, linha in bruto.iterrows():
        for dia in MAPA_DIAS.values():
            valor = linha[f"CHL TOTAL ({dia})"]
            linhas.append(
                {
                    "data_fisiologica": dia,
                    "genotipo": linha["Genótipo"],
                    "condicao": linha["Condição"],
                    "chl_total": pd.to_numeric(valor, errors="coerce"),
                }
            )
    return (
        pd.DataFrame(linhas)
        .dropna(subset=["chl_total"])
        .groupby(["data_fisiologica", "genotipo", "condicao"], as_index=False)
        .agg(chl_total=("chl_total", "mean"), n_repeticoes_fisiologia=("chl_total", "size"))
    )


def ler_espectros() -> tuple[pd.DataFrame, list[str]]:
    dados = pd.read_csv(ARQ_ESPECTRAL, sep=";")
    bandas = [str(i) for i in range(350, 2501) if str(i) in dados.columns]
    dados = dados.loc[dados["data_coleta"].isin(MAPA_DIAS) & dados["turno"].eq("manha")].copy()
    dados["data_fisiologica"] = dados["data_coleta"].map(MAPA_DIAS)
    dados["condicao"] = dados["condicao"].map({"IRRIG": "IRR", "NIRRIG": "NIR"})
    for banda in bandas:
        # A base CSV usa vírgula como separador decimal.
        dados[banda] = pd.to_numeric(
            dados[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    agrupado = (
        dados.groupby(["data_fisiologica", "genotipo", "condicao"], as_index=False)[bandas]
        .mean()
    )
    tamanhos = (
        dados.groupby(["data_fisiologica", "genotipo", "condicao"], as_index=False)
        .size()
        .rename(columns={"size": "n_curvas_espectrais"})
    )
    return agrupado.merge(tamanhos, on=["data_fisiologica", "genotipo", "condicao"]), bandas


def calcular_vip(modelo: PLSRegression) -> np.ndarray:
    """VIP clássico para resposta univariada."""
    t = modelo.x_scores_
    w = modelo.x_weights_
    q = modelo.y_loadings_.ravel()
    ssy = np.sum(t**2, axis=0) * q**2
    pesos = (w**2) / np.sum(w**2, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * (pesos @ ssy) / np.sum(ssy))


def melhor_n_componentes(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    # No LOO há n-1 linhas no treino; após centralização, no máximo n-2
    # componentes são informativos. Esse limite evita componentes degenerados.
    maximo = min(5, len(y) - 2, x.shape[1])
    resultados = []
    for n in range(1, maximo + 1):
        modelo = PLSRegression(n_components=n, scale=True)
        pred = cross_val_predict(modelo, x, y, cv=LeaveOneOut())
        resultados.append((n, float(np.sqrt(np.mean((y - pred.ravel()) ** 2)))))
    return min(resultados, key=lambda item: item[1])


def plotar(dia: str, tabela: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
    sig = tabela["significativa"]
    ax1.scatter(tabela.loc[~sig, "banda_nm"], tabela.loc[~sig, "rho_spearman"], s=7, c="#a0aec0", label="p ≥ 0,05")
    ax1.scatter(tabela.loc[sig, "banda_nm"], tabela.loc[sig, "rho_spearman"], s=8, c="#c53030", label="p < 0,05")
    ax1.axhline(0, color="#4a5568", linewidth=.8)
    ax1.set_ylabel("ρ de Spearman")
    ax1.legend(frameon=False, ncol=2)
    ax1.set_title(f"{dia}: curvas matinais × clorofila total (n = 6 grupos)")
    vip = tabela.dropna(subset=["vip"])
    ax2.plot(vip["banda_nm"], vip["vip"], color="#2b6cb0", linewidth=1.2)
    ax2.axhline(1, color="#c53030", linestyle="--", linewidth=1, label="VIP = 1")
    ax2.set(xlabel="Comprimento de onda (nm)", ylabel="VIP")
    ax2.legend(frameon=False)
    fig.savefig(SAIDA / f"{dia.replace('/', '-')}_spearman_vip.png", dpi=200)
    plt.close(fig)


def plotar_painel(tabelas: list[tuple[str, pd.DataFrame]]) -> None:
    """Painel único para comparar a resposta espectral entre as datas."""
    fig, eixos = plt.subplots(len(tabelas), 2, figsize=(15, 18), sharex=True, layout="constrained")
    for indice, (dia, tabela) in enumerate(tabelas):
        ax_rho, ax_vip = eixos[indice]
        significativa = tabela["significativa"]
        ax_rho.scatter(tabela.loc[~significativa, "banda_nm"], tabela.loc[~significativa, "rho_spearman"], s=3, c="#a0aec0")
        ax_rho.scatter(tabela.loc[significativa, "banda_nm"], tabela.loc[significativa, "rho_spearman"], s=4, c="#c53030")
        ax_rho.axhline(0, color="#4a5568", linewidth=.6)
        ax_rho.set_ylabel(f"{dia}\nρ")
        ax_rho.set_ylim(-1.08, 1.08)
        vip = tabela.dropna(subset=["vip"])
        if vip.empty:
            ax_vip.text(.5, .5, "Sem bandas com p < 0,05", ha="center", va="center", transform=ax_vip.transAxes, color="#4a5568")
        else:
            ax_vip.plot(vip["banda_nm"], vip["vip"], color="#2b6cb0", linewidth=1)
            ax_vip.axhline(1, color="#c53030", linestyle="--", linewidth=.8)
        ax_vip.set_ylabel("VIP")
    eixos[0, 0].set_title("Spearman: curva espectral × clorofila total\nVermelho: p < 0,05")
    eixos[0, 1].set_title("PLSR: importância da variável (VIP)\nLinha tracejada: VIP = 1")
    for eixo in eixos[-1]:
        eixo.set_xlabel("Comprimento de onda (nm)")
    fig.savefig(SAIDA / "painel_spearman_plsr_vip_por_dia.png", dpi=220)
    plt.close(fig)


def plotar_relacoes_top5(dia: str, dados: pd.DataFrame, top5: pd.DataFrame) -> None:
    """Dispersões finais: refletância nas bandas VIP × CHL Total."""
    if top5.empty:
        return
    fig, eixos = plt.subplots(1, len(top5), figsize=(4.2 * len(top5), 4.2), layout="constrained")
    eixos = np.atleast_1d(eixos)
    cores = {"IRR": "#2b6cb0", "NIR": "#c53030"}
    for eixo, (_, banda) in zip(eixos, top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        x = dados[coluna].to_numpy(float)
        y = dados["chl_total"].to_numpy(float)
        for _, ponto in dados.iterrows():
            eixo.scatter(
                ponto[coluna], ponto["chl_total"], s=48,
                color=cores[ponto["condicao"]], edgecolor="white", linewidth=.7,
            )
            eixo.annotate(str(ponto["genotipo"]), (ponto[coluna], ponto["chl_total"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
        inclinacao, intercepto = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        eixo.plot(xx, inclinacao * xx + intercepto, color="#4a5568", linewidth=1, linestyle="--")
        eixo.set_title(
            f"#{int(banda['rank_vip'])}: {coluna} nm\n"
            f"VIP={banda['vip']:.2f}; ρ={banda['rho_spearman']:.2f}; p={banda['p_valor']:.3f}",
            fontsize=9,
        )
        eixo.set(xlabel="Refletância", ylabel="CHL Total")
    fig.suptitle(f"{dia}: relação entre CHL Total e as bandas finais selecionadas", fontsize=13)
    fig.savefig(SAIDA / f"{dia.replace('/', '-')}_relacao_chl_total_top5_vip.png", dpi=220)
    plt.close(fig)


def analisar_dias_agrupados(dados: pd.DataFrame, bandas: list[str]) -> None:
    """Executa o mesmo fluxo sobre as 42 médias de todos os dias juntos."""
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    resultados = [spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({
        "banda_nm": np.array(bandas, dtype=int),
        "rho_spearman": [r[0] for r in resultados],
        "p_valor": [r[1] for r in resultados],
    })
    tabela["significativa"] = tabela["p_valor"] < 0.05
    selecionadas = tabela.loc[tabela["significativa"], "banda_nm"].astype(str).tolist()
    tabela["vip"] = np.nan
    if selecionadas:
        xs = dados[selecionadas].to_numpy(float)
        componentes, rmse_loo = melhor_n_componentes(xs, y)
        modelo = PLSRegression(n_components=componentes, scale=True).fit(xs, y)
        tabela.loc[tabela["significativa"], "vip"] = calcular_vip(modelo)
        r2_ajuste = float(modelo.score(xs, y))
        pred_ajuste = modelo.predict(xs).ravel()
        pred_loo = cross_val_predict(
            PLSRegression(n_components=componentes, scale=True), xs, y, cv=LeaveOneOut()
        ).ravel()
    else:
        componentes, rmse_loo, r2_ajuste = 0, np.nan, np.nan
    tabela.to_csv(SAIDA / "dias_agrupados_spearman_vip.csv", index=False, sep=";")
    top5 = tabela.dropna(subset=["vip"]).nlargest(5, "vip").copy()
    top5.insert(0, "analise", "dias_agrupados")
    top5.insert(1, "rank_vip", range(1, len(top5) + 1))
    top5["direcao_relacao"] = np.where(top5["rho_spearman"] > 0, "positiva", "negativa")
    top5.to_csv(SAIDA / "dias_agrupados_top5_vip_relacao_chl_total.csv", index=False, sep=";")
    pd.DataFrame([{
        "n_grupos_dia_genotipo_condicao": len(dados),
        "n_bandas_testadas": len(bandas),
        "n_bandas_spearman_p_lt_0_05": len(selecionadas),
        "componentes_plsr": componentes,
        "r2_ajuste_plsr": r2_ajuste,
        "rmse_loo_plsr": rmse_loo,
        "n_bandas_vip_ge_1": int((tabela["vip"] >= 1).sum()),
    }]).to_csv(SAIDA / "dias_agrupados_resumo.csv", index=False, sep=";")

    if selecionadas:
        predicoes = dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].rename(
            columns={"chl_total": "chl_total_observado"}
        )
        predicoes["chl_total_predito_ajuste"] = pred_ajuste
        predicoes["chl_total_predito_loo"] = pred_loo
        predicoes.to_csv(SAIDA / "predicoes_plsr_chl_total.csv", index=False, sep=";")
        fig, eixos = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        cores = plt.get_cmap("tab10")
        for eixo, previsto, titulo in (
            (eixos[0], pred_ajuste, "Ajuste"),
            (eixos[1], pred_loo, "Validação leave-one-out"),
        ):
            for indice, dia in enumerate(MAPA_DIAS.values()):
                parte = dados["data_fisiologica"].eq(dia)
                eixo.scatter(y[parte], previsto[parte], color=cores(indice), s=35, label=dia)
            limite_min = min(y.min(), previsto.min()) - .5
            limite_max = max(y.max(), previsto.max()) + .5
            eixo.plot([limite_min, limite_max], [limite_min, limite_max], "--", color="#4a5568", linewidth=1)
            eixo.set(xlim=(limite_min, limite_max), ylim=(limite_min, limite_max), xlabel="CHL Total observado", ylabel="CHL Total predito", title=titulo)
            eixo.text(
                .04, .94,
                f"R² = {r2_score(y, previsto):.3f}\nRMSE = {root_mean_squared_error(y, previsto):.2f}",
                transform=eixo.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.3"},
            )
        eixos[0].legend(title="Dia", frameon=False, fontsize=8, title_fontsize=9)
        fig.suptitle(f"PLSR para CHL Total — {len(selecionadas)} bandas Spearman (p < 0,05); {componentes} componentes")
        fig.savefig(SAIDA / "regressao_plsr_chl_total_observado_vs_predito.png", dpi=220)
        plt.close(fig)

    if top5.empty:
        return
    fig, eixos = plt.subplots(1, len(top5), figsize=(4.2 * len(top5), 4.1), layout="constrained")
    eixos = np.atleast_1d(eixos)
    cores = plt.get_cmap("tab10")
    ordem_dias = list(MAPA_DIAS.values())
    for eixo, (_, banda) in zip(eixos, top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        for indice, dia in enumerate(ordem_dias):
            parte = dados.loc[dados["data_fisiologica"].eq(dia)]
            eixo.scatter(parte[coluna], parte["chl_total"], s=26, color=cores(indice), label=dia)
        xx = np.linspace(dados[coluna].min(), dados[coluna].max(), 50)
        inclinacao, intercepto = np.polyfit(dados[coluna], y, 1)
        eixo.plot(xx, inclinacao * xx + intercepto, color="#4a5568", linewidth=1, linestyle="--")
        eixo.set_title(f"#{int(banda['rank_vip'])}: {coluna} nm\nVIP={banda['vip']:.2f}; ρ={banda['rho_spearman']:.2f}; p={banda['p_valor']:.3g}", fontsize=9)
        eixo.set(xlabel="Refletância", ylabel="CHL Total")
    eixos[0].legend(title="Dia", fontsize=7, title_fontsize=8, frameon=False)
    fig.suptitle("Dias agrupados: CHL Total × cinco bandas finais por VIP (n = 42)", fontsize=13)
    fig.savefig(SAIDA / "dias_agrupados_relacao_chl_total_top5_vip.png", dpi=220)
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    fisio = ler_fisiologia()
    espectros, bandas = ler_espectros()
    dados = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    dados.to_csv(SAIDA / "dados_agregados_genotipo_condicao.csv", index=False, sep=";")

    resumo: list[dict[str, object]] = []
    top_vip: list[pd.DataFrame] = []
    top5_vip: list[pd.DataFrame] = []
    tabelas: list[tuple[str, pd.DataFrame]] = []
    for dia in MAPA_DIAS.values():
        atual = dados.loc[dados["data_fisiologica"].eq(dia)].sort_values(["genotipo", "condicao"])
        if len(atual) != 6:
            raise ValueError(f"{dia}: esperados 6 grupos, encontrados {len(atual)}")
        y = atual["chl_total"].to_numpy(float)
        x = atual[bandas].to_numpy(float)
        resultados = [spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
        tabela = pd.DataFrame(
            {
                "banda_nm": np.array(bandas, dtype=int),
                "rho_spearman": [r[0] for r in resultados],
                "p_valor": [r[1] for r in resultados],
            }
        )
        tabela["significativa"] = tabela["p_valor"] < 0.05
        selecionadas = tabela.loc[tabela["significativa"], "banda_nm"].astype(str).tolist()
        tabela["vip"] = np.nan
        if selecionadas:
            xs = atual[selecionadas].to_numpy(float)
            componentes, rmse_loo = melhor_n_componentes(xs, y)
            modelo = PLSRegression(n_components=componentes, scale=True).fit(xs, y)
            tabela.loc[tabela["significativa"], "vip"] = calcular_vip(modelo)
            r2_ajuste = float(modelo.score(xs, y))
        else:
            componentes, rmse_loo, r2_ajuste = 0, np.nan, np.nan
        tabela.to_csv(SAIDA / f"{dia.replace('/', '-')}_spearman_vip.csv", index=False, sep=";")
        top = tabela.dropna(subset=["vip"]).nlargest(10, "vip").copy()
        top.insert(0, "data_fisiologica", dia)
        top_vip.append(top)
        top5 = top.head(5).copy()
        top5.insert(1, "rank_vip", range(1, len(top5) + 1))
        top5["direcao_relacao"] = np.where(top5["rho_spearman"] > 0, "positiva", "negativa")
        top5_vip.append(top5)
        tabelas.append((dia, tabela))
        plotar(dia, tabela)
        plotar_relacoes_top5(dia, atual, top5)
        resumo.append(
            {
                "data_fisiologica": dia,
                "n_grupos": len(atual),
                "n_bandas_testadas": len(bandas),
                "n_bandas_spearman_p_lt_0_05": len(selecionadas),
                "componentes_plsr": componentes,
                "r2_ajuste_plsr": r2_ajuste,
                "rmse_loo_plsr": rmse_loo,
                "n_bandas_vip_ge_1": int((tabela["vip"] >= 1).sum()),
            }
        )
    pd.DataFrame(resumo).to_csv(SAIDA / "resumo_por_dia.csv", index=False, sep=";")
    pd.concat(top_vip, ignore_index=True).to_csv(SAIDA / "top10_vip_por_dia.csv", index=False, sep=";")
    pd.concat(top5_vip, ignore_index=True).to_csv(SAIDA / "top5_vip_relacao_chl_total.csv", index=False, sep=";")
    plotar_painel(tabelas)
    analisar_dias_agrupados(dados, bandas)


if __name__ == "__main__":
    main()
