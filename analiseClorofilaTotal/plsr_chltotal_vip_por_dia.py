#!/usr/bin/env python3
"""CHL Total: seleção VIP independente em cada data, com controle de redundância.

Cada data passa por: pré-processamento espectral, agrupamento local de bandas
redundantes por Spearman, PLSR/VIP próprio e seleção gulosa das cinco bandas
de maior VIP que mantenham |r de Pearson| < 0,80 entre si.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))
sys.path.insert(0, str(ROOT / "analiseClorofilaTotal"))
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402
import plsr_chltotal_global_top5_por_dia as base  # noqa: E402

OUT = ROOT / "analiseClorofilaTotal" / "resultados" / "plsr_vip_por_dia_top5"
ALVO_NOME = "CHL Total"
ALVO_CHAVE = "chl_total"
LIMIAR_ABS_CORRELACAO = .80
N_TOP = 5


def vip(modelo):
    """Variable Importance in Projection para PLS com uma variável resposta."""
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * ((w_norm ** 2) * ssy).sum(axis=1) / ssy.sum())


def selecionar_por_dia(X, y, bandas, blocos):
    """Retorna Top 5 VIP do dia após as duas etapas de redução de redundância."""
    corr_s = spearman_matriz(X)
    grupos_s = agrupar(corr_s, bandas, limiar_r=.80, janela_nm=10.0)
    rho_y = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = np.array([
        np.flatnonzero(grupos_s == g)[np.nanargmax(np.abs(rho_y[grupos_s == g]))]
        for g in np.unique(grupos_s)
    ])
    Xr, br = X[:, reps], bandas[reps]
    n_inicial, tab_inicial = base.cv_componentes(Xr, y, blocos)
    modelo_inicial = PLSRegression(n_components=n_inicial, scale=True).fit(Xr, y)
    vip_inicial = vip(modelo_inicial)
    corr_p = np.corrcoef(Xr, rowvar=False)
    ranking = pd.DataFrame({
        "indice_representante": np.arange(len(br)), "banda_nm": br,
        "vip_selecao": vip_inicial, f"rho_spearman_{ALVO_CHAVE}": rho_y[reps],
    }).sort_values("vip_selecao", ascending=False).reset_index(drop=True)
    ranking.insert(0, "ranking_vip_selecao", np.arange(1, len(ranking) + 1))

    escolhidas, max_corr = [], []
    for _, linha in ranking.iterrows():
        indice = int(linha.indice_representante)
        maior = max((abs(corr_p[indice, anterior]) for anterior in escolhidas), default=np.nan)
        max_corr.append(maior)
        if (not escolhidas or maior < LIMIAR_ABS_CORRELACAO) and len(escolhidas) < N_TOP:
            escolhidas.append(indice)
    if len(escolhidas) < N_TOP:
        raise ValueError("Não foi possível selecionar cinco bandas VIP não colineares.")
    ranking["max_abs_r_pearson_com_banda_ja_selecionada"] = max_corr
    ranking["selecionada_top5_nao_colinear"] = ranking.indice_representante.isin(escolhidas)

    top = ranking[ranking.selecionada_top5_nao_colinear].copy()
    top = top.sort_values("ranking_vip_selecao").reset_index(drop=True)
    bandas_top = top.banda_nm.to_numpy(int)
    indices_top = np.array([np.where(bandas == banda)[0][0] for banda in bandas_top])
    Xtop = X[:, indices_top]
    n_final, tab_final = base.cv_componentes(Xtop, y, blocos)
    modelo_final = PLSRegression(n_components=n_final, scale=True).fit(Xtop, y)
    top["vip_modelo_final"] = vip(modelo_final)
    top.insert(0, "ranking_top5_selecao", np.arange(1, len(top) + 1))
    corr_top = corr_p[np.ix_(top.indice_representante.to_numpy(int), top.indice_representante.to_numpy(int))]
    return top, ranking, pd.DataFrame(corr_top, index=bandas_top, columns=bandas_top), tab_inicial, tab_final, modelo_final, Xtop


def plotar_bandas(consolidado, caminho):
    """Mostra todas as cinco bandas VIP selecionadas em cada uma das sete datas."""
    ordem = [dados[0] for dados in base.MAPA.values()]
    posicao = {data: i for i, data in enumerate(ordem)}
    fig, ax = plt.subplots(figsize=(14, 6.5), layout="constrained")
    pontos = ax.scatter(consolidado.banda_nm, consolidado.data_chl.map(posicao),
                        c=consolidado.vip_modelo_final, cmap="viridis", s=115,
                        edgecolor="#202020", linewidth=.45, zorder=3)
    # Desloca rótulos de bandas próximas para preservar a leitura no mesmo dia.
    for data, grupo in consolidado.groupby("data_chl", sort=False):
        ordenado = grupo.sort_values("banda_nm")
        blocos, atual = [], []
        anterior = None
        for _, linha in ordenado.iterrows():
            if anterior is not None and linha.banda_nm - anterior > 45:
                blocos.append(atual); atual = []
            atual.append(linha)
            anterior = linha.banda_nm
        blocos.append(atual)
        for bloco in blocos:
            for indice, linha in enumerate(bloco):
                dx = (indice - (len(bloco) - 1) / 2) * 18
                dy = 8 + (indice % 2) * 10
                ax.annotate(f"{int(linha.banda_nm)}", (linha.banda_nm, posicao[data]),
                            xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set(yticks=np.arange(len(ordem)), yticklabels=ordem, xlabel="Comprimento de onda (nm)",
           ylabel=f"Data de {ALVO_NOME}", xlim=(370, 2480),
           title=f"{ALVO_NOME} — Top 5 bandas VIP não colineares por data")
    ax.grid(axis="x", color="#d9d9d9", linewidth=.7)
    ax.invert_yaxis()
    fig.colorbar(pontos, ax=ax, label="VIP do modelo final da data")
    fig.savefig(caminho, dpi=260)
    plt.close(fig)


def plotar_predicoes(dados, observado, predito, data, caminho):
    """Observado × predito em CV, colorido pela condição hídrica."""
    cores = {"IRRIG": "#1976d2", "NIRRIG": "#d32f2f"}
    rotulos = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
    fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    for condicao in ("IRRIG", "NIRRIG"):
        mascara = dados.condicao.eq(condicao).to_numpy()
        ax.scatter(observado[mascara], predito[mascara], s=62, color=cores[condicao],
                   edgecolor="white", linewidth=.7, label=rotulos[condicao], zorder=3)
    minimo, maximo = min(observado.min(), predito.min()), max(observado.max(), predito.max())
    margem = max((maximo - minimo) * .08, .2)
    ax.plot([minimo-margem, maximo+margem], [minimo-margem, maximo+margem], "--",
            color="#4a5568", lw=1.2, label="1:1", zorder=1)
    ax.text(.04, .96, f"R² CV = {r2_score(observado, predito):.3f}\nRMSE CV = {mean_squared_error(observado, predito)**.5:.3f}",
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.35"})
    ax.set(xlim=(minimo-margem, maximo+margem), ylim=(minimo-margem, maximo+margem),
           xlabel=f"{ALVO_NOME} observado", ylabel=f"{ALVO_NOME} predito (CV por bloco)",
           title=f"{ALVO_NOME} — {data}\nPLSR com Top 5 VIP da data")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(caminho, dpi=240)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dados, bandas = base.carregar()
    resultados, bandas_todas = [], []
    for chave, (data, _, coleta) in base.MAPA.items():
        dia = dados[dados.chave_data.eq(chave)].reset_index(drop=True)
        X = dia[[str(b) for b in bandas]].to_numpy(float)
        y, blocos = dia.chl_total.to_numpy(float), dia.bloco.to_numpy()
        top, ranking, corr_top, tab_inicial, tab_final, modelo, Xtop = selecionar_por_dia(X, y, bandas, blocos)
        ajuste = modelo.predict(Xtop).ravel()
        cv = base.predicao_cv(Xtop, y, blocos, modelo.n_components)
        top.insert(0, "data_chl", data); top.insert(1, "coleta_espectral", coleta)
        ranking.insert(0, "data_chl", data); ranking.insert(1, "coleta_espectral", coleta)
        top.to_csv(OUT / f"{chave}_top5_vip_nao_colinear.csv", sep=";", index=False)
        ranking.to_csv(OUT / f"{chave}_ranking_vip_representantes.csv", sep=";", index=False)
        corr_top.rename_axis("banda_nm").to_csv(OUT / f"{chave}_correlacao_pearson_top5.csv", sep=";")
        tab_inicial.to_csv(OUT / f"{chave}_componentes_cv_inicial.csv", sep=";", index=False)
        tab_final.to_csv(OUT / f"{chave}_componentes_cv_top5.csv", sep=";", index=False)
        dia[["bloco", "genotipo", "condicao", "data_chl", "chl_total"]].assign(
            predito_ajuste=ajuste, predito_cv_bloco=cv, residuo_cv_bloco=y-cv
        ).to_csv(OUT / f"{chave}_predicoes_top5_vip.csv", sep=";", index=False)
        plotar_predicoes(dia, y, cv, data, OUT / f"{chave}_observado_vs_predito_cv_top5_vip.png")
        resultados.append({"data_chl": data, "coleta_espectral": coleta, "n": len(y),
                           "representantes_spearman": len(ranking), "componentes_top5": modelo.n_components,
                           "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": mean_squared_error(y, ajuste)**.5,
                           "r2_cv_bloco": r2_score(y, cv), "rmse_cv_bloco": mean_squared_error(y, cv)**.5})
        bandas_todas.append(top)
    consolidado = pd.concat(bandas_todas, ignore_index=True)
    resumo = pd.DataFrame(resultados)
    consolidado.to_csv(OUT / "top5_vip_por_dia.csv", sep=";", index=False)
    resumo.to_csv(OUT / "resumo_metricas_por_dia.csv", sep=";", index=False)
    plotar_bandas(consolidado, OUT / "bandas_top5_vip_por_dia.png")
    texto = [f"# {ALVO_NOME} — PLSR/VIP por data", "",
             f"Cada uma das {len(base.MAPA)} datas foi analisada de forma independente. Antes do ajuste, os espectros receberam correção de emendas, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. Bandas localmente redundantes foram agrupadas por Spearman |ρ| > 0,80 em janela <10 nm, mantendo-se a mais associada a {ALVO_NOME} em cada grupo.", "",
             "O PLSR inicial de cada data definiu o VIP. As cinco bandas foram escolhidas em ordem de VIP, mas uma candidata só foi aceita se |r de Pearson| <0,80 contra todas as já selecionadas. O PLSR final e as predições CV por bloco usam somente essas cinco bandas.", "",
             "## Métricas por data", "", base.tabela_markdown(resumo), "",
             f"O gráfico `bandas_top5_vip_por_dia.png` concentra as {len(consolidado)} bandas selecionadas. A cor indica o VIP recalculado no modelo final de cada data."]
    (OUT / "relatorio.md").write_text("\n".join(texto) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
