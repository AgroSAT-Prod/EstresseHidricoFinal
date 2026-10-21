#!/usr/bin/env python3
"""PLSR global de CRA e predição diária com as cinco bandas globais.

O ranking é obtido em um único PLSR com todas as datas de CRA reunidas. As
bandas são previamente reduzidas por colinearidade espectral local (Spearman
|rho| > 0,80, janela < 10 nm); assim, bandas vizinhas praticamente idênticas
não ocupam simultaneamente o Top 5. O ranking final usa o valor absoluto do
coeficiente PLSR padronizado. Depois, exatamente as cinco bandas globais são
usadas para ajustar e validar um PLSR independente em cada data.

Antes da seleção, cada espectro passa pela pipeline usada no projeto:
correção dos saltos entre detectores, remoção das bordas (400--2450 nm),
suavização Savitzky--Golay (janela 11, polinômio 2) e SNV.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))
sys.path.insert(0, str(ROOT / "preprocessamento_espectral"))
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402
from normalizacao import snv  # noqa: E402
from preprocessamento import preprocessar  # noqa: E402
from suavizacao import savitzky_golay  # noqa: E402

FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
OUT = ROOT / "analiseCRA" / "resultados" / "plsr_global_top5_por_dia"
MAPA = {
    "24_02": ("24/02", "CRA (24/02)", "D03M"),
    "02_03": ("02/03", "CRA (02/03)", "D09M"),
    "03_03": ("03/03", "CRA (03/03)", "D10M"),
}
MAX_COMPONENTES = 5
LIMIAR_ABS_CORRELACAO = 0.80


def cv_componentes(X, y, grupos):
    folds = list(GroupKFold(n_splits=len(np.unique(grupos))).split(X, y, grupos))
    limite = min(MAX_COMPONENTES, X.shape[1], min(len(tr) - 1 for tr, _ in folds))
    linhas = []
    for n in range(1, limite + 1):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
        linhas.append({"n_componentes": n, "r2_cv_bloco": r2_score(y, pred),
                       "rmse_cv_bloco": mean_squared_error(y, pred) ** .5})
    tabela = pd.DataFrame(linhas)
    return int(tabela.loc[tabela.rmse_cv_bloco.idxmin(), "n_componentes"]), tabela


def predicao_cv(X, y, grupos, n):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=len(np.unique(grupos))).split(X, y, grupos):
        pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
    return pred


def coeficientes_padronizados(modelo):
    """Coeficientes para X e y em unidades de desvio-padrão (comparáveis)."""
    # ``coef_`` do scikit-learn está nas unidades originais de y por unidade
    # original de X. Multiplicar por sd(X)/sd(y) fornece beta padronizado.
    bruto = np.asarray(modelo.coef_).reshape(-1)
    return bruto * modelo._x_std / modelo._y_std[0]


def tabela_markdown(df, casas=4):
    """Tabela Markdown sem depender do pacote opcional ``tabulate``."""
    cab = [str(c) for c in df.columns]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "|".join(["---"] * len(cab)) + "|"]
    for _, linha in df.iterrows():
        valores = []
        for valor in linha:
            valores.append(f"{valor:.{casas}f}" if isinstance(valor, (float, np.floating)) else str(valor))
        linhas.append("| " + " | ".join(valores) + " |")
    return "\n".join(linhas)


def selecionar_top5_nao_colinear(ranking, X_representantes, n_bandas=5):
    """Seleciona por coeficiente PLSR, barrando pares com |r Pearson| >= 0,80."""
    corr = np.corrcoef(X_representantes, rowvar=False)
    selecionados, maximos = [], []
    for pos in range(len(ranking)):
        maior = max((abs(corr[pos, outro]) for outro in selecionados), default=np.nan)
        maximos.append(maior)
        if (not selecionados or maior < LIMIAR_ABS_CORRELACAO) and len(selecionados) < n_bandas:
            selecionados.append(pos)
    if len(selecionados) < n_bandas:
        raise ValueError("Não foi possível selecionar cinco bandas não colineares.")
    # ``ranking`` ainda está na mesma ordem das colunas de X_representantes
    # até o sort abaixo; o índice original preserva essa correspondência.
    return selecionados, corr


def plotar_predicoes_diarias(dados, observado, predito, data, caminho):
    """CRA observado × predito em CV, com cores fixas por condição hídrica."""
    cores = {"IRRIG": "#1976d2", "NIRRIG": "#d32f2f"}
    rotulos = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
    fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    for condicao in ("IRRIG", "NIRRIG"):
        mask = dados.condicao.eq(condicao).to_numpy()
        ax.scatter(observado[mask], predito[mask], s=62, color=cores[condicao],
                   edgecolor="white", linewidth=.7, label=rotulos[condicao], zorder=3)
    minimo, maximo = min(observado.min(), predito.min()), max(observado.max(), predito.max())
    margem = max((maximo - minimo) * .08, .5)
    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem],
            "--", color="#4a5568", lw=1.2, label="1:1", zorder=1)
    r2 = r2_score(observado, predito)
    rmse = mean_squared_error(observado, predito) ** .5
    ax.text(.04, .96, f"R² CV = {r2:.3f}\nRMSE CV = {rmse:.3f}", transform=ax.transAxes,
            va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.35"})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel="CRA observado", ylabel="CRA predito (CV por bloco)",
           title=f"CRA — {data}\nPLSR com as 5 bandas globais")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(caminho, dpi=240)
    plt.close(fig)


def carregar():
    fis = pd.read_excel(FISIO)
    fis["condicao"] = fis["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["genotipo"] = fis["Genótipo"]
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda x: f"B{x + 1}")
    esp_bruto = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    # Pré-processamento aplicado por espectro, antes da agregação por bloco.
    esp_corrigido, w = preprocessar(esp_bruto)
    bandas = w.astype(int)
    colunas = [str(b) for b in bandas]
    X = esp_corrigido[colunas].to_numpy(float)
    X = snv(savitzky_golay(X, window_length=11, polyorder=2))
    meta_espectral = esp_corrigido[[c for c in esp_corrigido.columns if not c.isdigit()]].reset_index(drop=True)
    esp = pd.concat([meta_espectral, pd.DataFrame(X, columns=colunas)], axis=1)
    meta = ["bloco", "genotipo", "condicao"]
    partes = []
    for chave, (data, alvo, coleta) in MAPA.items():
        medias = (esp[(esp.data_coleta == coleta) & (esp.turno == "manha")]
                  .groupby(meta, as_index=False)[[str(b) for b in bandas]].mean())
        parte = fis[meta + [alvo]].merge(medias, on=meta, how="inner", validate="one_to_one")
        parte = parte.dropna(subset=[alvo]).rename(columns={alvo: "cra"})
        parte.insert(0, "chave_data", chave)
        parte.insert(1, "data_cra", data)
        partes.append(parte)
    return pd.concat(partes, ignore_index=True), bandas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dados, bandas = carregar()
    colunas = [str(b) for b in bandas]
    X = dados[colunas].to_numpy(float)
    y = dados.cra.to_numpy(float)
    grupos = dados.bloco.to_numpy()

    # Uma representante por pequeno grupo de bandas localmente redundantes.
    corr = spearman_matriz(X)
    grupos_rho = agrupar(corr, bandas, limiar_r=.80, janela_nm=10.0)
    rho_y = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = np.array([np.flatnonzero(grupos_rho == g)[np.nanargmax(np.abs(rho_y[grupos_rho == g]))]
                     for g in np.unique(grupos_rho)])
    Xr, br = X[:, reps], bandas[reps]

    n_global, tab_global = cv_componentes(Xr, y, grupos)
    modelo_global = PLSRegression(n_components=n_global, scale=True).fit(Xr, y)
    coef_bruto = np.asarray(modelo_global.coef_).reshape(-1)
    coef = coeficientes_padronizados(modelo_global)
    ranking = pd.DataFrame({
        "indice_representante": np.arange(len(br)),
        "banda_nm": br,
        "coeficiente_plsr_padronizado": coef,
        "coeficiente_plsr_absoluto": np.abs(coef),
        "coeficiente_plsr_bruto": coef_bruto,
        "rho_spearman_cra": rho_y[reps],
    }).sort_values("coeficiente_plsr_absoluto", ascending=False).reset_index(drop=True)
    ranking.insert(0, "ranking_coeficiente_plsr", np.arange(1, len(ranking) + 1))
    # Exclusão final de multicolinearidade: a seleção gulosa percorre o
    # ranking PLSR e só mantém uma banda se for independente das já aceitas.
    corr_reps = np.corrcoef(Xr, rowvar=False)
    escolhidas = []
    max_corr = []
    for _, linha in ranking.iterrows():
        indice = int(linha.indice_representante)
        maior = max((abs(corr_reps[indice, outra]) for outra in escolhidas), default=np.nan)
        max_corr.append(maior)
        if (not escolhidas or maior < LIMIAR_ABS_CORRELACAO) and len(escolhidas) < 5:
            escolhidas.append(indice)
    if len(escolhidas) < 5:
        raise ValueError("Não foi possível selecionar cinco bandas não colineares (|r| < 0,80).")
    ranking["max_abs_r_pearson_com_banda_ja_selecionada"] = max_corr
    ranking["selecionada_top5_nao_colinear"] = ranking.indice_representante.isin(escolhidas)
    top5 = ranking[ranking.selecionada_top5_nao_colinear].copy()
    top5 = top5.sort_values("ranking_coeficiente_plsr").reset_index(drop=True)
    top5.insert(0, "ranking_top5_nao_colinear", np.arange(1, len(top5) + 1))
    top_bandas = top5.banda_nm.to_numpy(int)
    top_idx = np.array([np.where(bandas == b)[0][0] for b in top_bandas])
    ranking.to_csv(OUT / "ranking_global_coeficientes_plsr.csv", sep=";", index=False)
    top5.to_csv(OUT / "top5_bandas_globais_plsr.csv", sep=";", index=False)
    top_indices = top5.indice_representante.to_numpy(int)
    matriz_top5 = pd.DataFrame(corr_reps[np.ix_(top_indices, top_indices)], index=top_bandas, columns=top_bandas)
    matriz_top5.index.name = "banda_nm"
    matriz_top5.to_csv(OUT / "correlacao_pearson_top5_nao_colinear.csv", sep=";")
    tab_global.to_csv(OUT / "global_componentes_cv.csv", sep=";", index=False)

    Xtop = X[:, top_idx]
    n_top5, tab_top5 = cv_componentes(Xtop, y, grupos)
    modelo_top5 = PLSRegression(n_components=n_top5, scale=True).fit(Xtop, y)
    ajuste_top5 = modelo_top5.predict(Xtop).ravel()
    pred_top5 = predicao_cv(Xtop, y, grupos, n_top5)
    tab_top5.to_csv(OUT / "global_top5_componentes_cv.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", "data_cra", "cra"]].assign(
        predito_ajuste=ajuste_top5, predito_cv_bloco=pred_top5, residuo_cv_bloco=y-pred_top5
    ).to_csv(OUT / "global_top5_predicoes.csv", sep=";", index=False)
    resumo = [{"escopo": "Global (todas as datas; Top 5)", "n": len(y), "componentes": n_top5,
               "r2_ajuste": r2_score(y, ajuste_top5),
               "rmse_ajuste": mean_squared_error(y, ajuste_top5) ** .5,
               "r2_cv_bloco": r2_score(y, pred_top5),
               "rmse_cv_bloco": mean_squared_error(y, pred_top5) ** .5}]

    for chave, (data, _, coleta) in MAPA.items():
        dia = dados[dados.chave_data == chave].reset_index(drop=True)
        xd, yd, gd = dia[[str(b) for b in top_bandas]].to_numpy(float), dia.cra.to_numpy(float), dia.bloco.to_numpy()
        n, tabela = cv_componentes(xd, yd, gd)
        modelo = PLSRegression(n_components=n, scale=True).fit(xd, yd)
        ajuste, cv = modelo.predict(xd).ravel(), predicao_cv(xd, yd, gd, n)
        tabela.to_csv(OUT / f"{chave}_componentes_cv.csv", sep=";", index=False)
        dia[["bloco", "genotipo", "condicao", "data_cra", "cra"]].assign(
            predito_ajuste=ajuste, predito_cv_bloco=cv, residuo_cv_bloco=yd-cv
        ).to_csv(OUT / f"{chave}_predicoes_top5.csv", sep=";", index=False)
        plotar_predicoes_diarias(dia, yd, cv, data, OUT / f"{chave}_observado_vs_predito_cv_top5.png")
        resumo.append({"escopo": data, "coleta_espectral": coleta, "n": len(yd), "componentes": n,
                       "r2_ajuste": r2_score(yd, ajuste), "rmse_ajuste": mean_squared_error(yd, ajuste) ** .5,
                       "r2_cv_bloco": r2_score(yd, cv), "rmse_cv_bloco": mean_squared_error(yd, cv) ** .5})

    resumo = pd.DataFrame(resumo)
    resumo.to_csv(OUT / "resumo_metricas.csv", sep=";", index=False)
    linhas = ["# PLSR — CRA global e predição por data", "",
              f"Pré-processamento aplicado antes da agregação por bloco: correção de emendas dos detectores, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. O ranking foi calculado no PLSR único das três datas. O coeficiente é padronizado; sinal positivo indica aumento previsto de CRA quando a reflectância da banda aumenta, mantendo as demais bandas constantes. A ordem usa o módulo do coeficiente. Antes do Top 5, excluíram-se bandas multicolineares: uma candidata só foi mantida se |r de Pearson| < {LIMIAR_ABS_CORRELACAO:.2f} contra cada banda já selecionada.", "",
              "## Top 5 global", "", tabela_markdown(top5), "",
              "## Métricas", "", tabela_markdown(resumo), "",
              "As predições diárias usam somente estas cinco bandas. Componentes foram escolhidos pelo menor RMSE em GroupKFold por bloco (B1--B4). As métricas de CV são diagnósticas: o Top 5 foi selecionado no conjunto global antes da CV."]
    (OUT / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
