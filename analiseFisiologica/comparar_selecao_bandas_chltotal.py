#!/usr/bin/env python3
"""Compara quatro formas de diversificar bandas para CHL Total agrupado.

Base comum: bandas retidas por Spearman (p < 0,05) e seus VIPs do PLSR
agrupado. Todas as opções retornam até cinco bandas.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cross_decomposition import PLSRegression

import clorofila_total_spearman_plsr as base


SAIDA = base.SAIDA
LARGURA_JANELA_NM = 20
LIMIAR_COLINEARIDADE = 0.95
N_FINAL = 5


def adicionar_metodo(nome: str, tabela: pd.DataFrame, extra: dict[int, dict[str, object]] | None = None) -> pd.DataFrame:
    resultado = tabela.copy().reset_index(drop=True)
    resultado.insert(0, "metodo", nome)
    resultado.insert(1, "rank", range(1, len(resultado) + 1))
    resultado["direcao_relacao"] = np.where(resultado["rho_spearman"] > 0, "positiva", "negativa")
    if extra:
        for coluna in {chave for valores in extra.values() for chave in valores}:
            resultado[coluna] = [extra.get(int(b), {}).get(coluna, "") for b in resultado["banda_nm"]]
    return resultado


def janela_minima(vip: pd.DataFrame) -> pd.DataFrame:
    """Maior VIP disponível, depois a próxima banda distante ao menos 20 nm."""
    escolhidas: list[pd.Series] = []
    for _, linha in vip.sort_values("vip", ascending=False).iterrows():
        if all(abs(linha["banda_nm"] - item["banda_nm"]) >= LARGURA_JANELA_NM for item in escolhidas):
            escolhidas.append(linha)
        if len(escolhidas) == N_FINAL:
            break
    return pd.DataFrame(escolhidas)


def representantes_colinearidade(vip: pd.DataFrame, dados: pd.DataFrame) -> pd.DataFrame:
    """Componentes conectados de |correlação entre bandas| >= 0,95; maior VIP por grupo."""
    bandas = vip["banda_nm"].astype(int).tolist()
    matriz = dados[[str(b) for b in bandas]].corr(method="pearson").abs().to_numpy()
    pai = list(range(len(bandas)))

    def raiz(indice: int) -> int:
        while pai[indice] != indice:
            pai[indice] = pai[pai[indice]]
            indice = pai[indice]
        return indice

    def unir(a: int, b: int) -> None:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            pai[rb] = ra

    for i in range(len(bandas)):
        for j in range(i + 1, len(bandas)):
            if matriz[i, j] >= LIMIAR_COLINEARIDADE:
                unir(i, j)
    grupos: dict[int, list[int]] = defaultdict(list)
    for indice in range(len(bandas)):
        grupos[raiz(indice)].append(indice)
    candidatos = []
    extra: dict[int, dict[str, object]] = {}
    for membros in grupos.values():
        parte = vip.iloc[membros]
        melhor = parte.loc[parte["vip"].idxmax()]
        candidatos.append(melhor)
        extra[int(melhor["banda_nm"])] = {"tamanho_grupo_colinear": len(membros)}
    saida = pd.DataFrame(candidatos).sort_values("vip", ascending=False).head(N_FINAL)
    return adicionar_metodo("grupo_colinear_r_ge_0_95", saida, extra)


def interval_pls(vip: pd.DataFrame, dados: pd.DataFrame) -> pd.DataFrame:
    """iPLS: cada janela de 20 nm é avaliada pelo RMSE leave-one-out."""
    registros = []
    for inicio in range(350, 2501, LARGURA_JANELA_NM):
        fim = inicio + LARGURA_JANELA_NM - 1
        bandas = vip.loc[vip["banda_nm"].between(inicio, fim), "banda_nm"].astype(int).tolist()
        if not bandas:
            continue
        x = dados[[str(b) for b in bandas]].to_numpy(float)
        y = dados["chl_total"].to_numpy(float)
        componentes, rmse = base.melhor_n_componentes(x, y)
        modelo = PLSRegression(n_components=componentes, scale=True).fit(x, y)
        vip_intervalo = base.calcular_vip(modelo)
        indice = int(np.argmax(vip_intervalo))
        banda = bandas[indice]
        linha = vip.loc[vip["banda_nm"].eq(banda)].iloc[0].copy()
        linha["vip_intervalo"] = vip_intervalo[indice]
        linha["rmse_loo_intervalo"] = rmse
        linha["intervalo_nm"] = f"{inicio}-{fim}"
        registros.append(linha)
    melhores = pd.DataFrame(registros).sort_values("rmse_loo_intervalo").head(N_FINAL)
    # A construção a partir de Series pode promover a coluna para float; bandas
    # são comprimentos de onda inteiros e também nomes de colunas na matriz.
    melhores["banda_nm"] = melhores["banda_nm"].astype(int)
    return adicionar_metodo("iPLS_intervalos_20_nm", melhores)


def picos_locais(vip: pd.DataFrame) -> pd.DataFrame:
    """Picos locais de VIP, exigindo separação mínima de 20 nm."""
    grade = np.arange(350, 2501)
    valores = np.zeros(len(grade))
    posicao = {int(b): i for i, b in enumerate(grade)}
    for _, linha in vip.iterrows():
        valores[posicao[int(linha["banda_nm"])]] = linha["vip"]
    picos, _ = find_peaks(valores, distance=LARGURA_JANELA_NM)
    bandas_pico = set(grade[picos])
    # Mantém também extremidades caso sejam máximos de uma região significativa.
    bandas_pico.update([int(vip.loc[vip["vip"].idxmax(), "banda_nm"])])
    melhores = vip.loc[vip["banda_nm"].isin(bandas_pico)].nlargest(N_FINAL, "vip")
    return adicionar_metodo("picos_locais_vip", melhores)


def plotar(vip: pd.DataFrame, selecoes: pd.DataFrame, parametro: str = "CHL Total") -> None:
    metodos = list(selecoes["metodo"].unique())
    fig, eixos = plt.subplots(len(metodos), 1, figsize=(14, 9), sharex=True, layout="constrained")
    for eixo, metodo in zip(eixos, metodos):
        eixo.plot(vip["banda_nm"], vip["vip"], color="#a0aec0", linewidth=.8, label="VIP PLSR")
        parte = selecoes.loc[selecoes["metodo"].eq(metodo)]
        eixo.scatter(parte["banda_nm"], parte["vip"], color="#c53030", s=36, zorder=3, label="selecionada")
        for _, linha in parte.iterrows():
            eixo.annotate(str(int(linha["banda_nm"])), (linha["banda_nm"], linha["vip"]), xytext=(3, 4), textcoords="offset points", fontsize=8)
        eixo.axhline(1, color="#2b6cb0", linestyle="--", linewidth=.8)
        eixo.set(ylabel="VIP", title=metodo.replace("_", " "))
        eixo.legend(frameon=False, loc="upper right")
    eixos[-1].set_xlabel("Comprimento de onda (nm)")
    fig.suptitle(f"Dias agrupados: estratégias para selecionar bandas — {parametro}", fontsize=14)
    fig.savefig(SAIDA / "comparacao_quatro_metodos_selecao_bandas.png", dpi=220)
    plt.close(fig)


def main() -> None:
    fisio = base.ler_fisiologia()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.array(bandas, int), "rho_spearman": [r[0] for r in correlacoes], "p_valor": [r[1] for r in correlacoes]})
    retidas = tabela.loc[tabela["p_valor"] < .05].copy()
    xr = dados[retidas["banda_nm"].astype(str)].to_numpy(float)
    componentes, _ = base.melhor_n_componentes(xr, y)
    modelo = PLSRegression(n_components=componentes, scale=True).fit(xr, y)
    retidas["vip"] = base.calcular_vip(modelo)

    selecoes = [
        adicionar_metodo("janela_minima_20_nm", janela_minima(retidas)),
        representantes_colinearidade(retidas, dados),
        interval_pls(retidas, dados),
        picos_locais(retidas),
    ]
    resultado = pd.concat(selecoes, ignore_index=True)
    resultado.to_csv(SAIDA / "comparacao_quatro_metodos_bandas_selecionadas.csv", index=False, sep=";")
    plotar(retidas, resultado)


if __name__ == "__main__":
    main()
