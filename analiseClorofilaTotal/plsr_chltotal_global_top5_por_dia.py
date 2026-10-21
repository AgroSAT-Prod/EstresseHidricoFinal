#!/usr/bin/env python3
"""CHL Total: PLSR global, Top 5 não colinear e predições por data.

Pipeline idêntica à de CRA: correção de emendas, recorte 400--2450 nm,
Savitzky--Golay (11, 2), SNV, redução local por Spearman, ranking pelo
coeficiente PLSR padronizado e exclusão final de pares |r de Pearson| >= .80.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "reducaoColinearidade"), str(ROOT / "preprocessamento_espectral")])
from normalizacao import snv  # noqa: E402
from preprocessamento import preprocessar  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402
from suavizacao import savitzky_golay  # noqa: E402

FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
OUT = ROOT / "analiseClorofilaTotal" / "resultados" / "plsr_global_top5_por_dia"
ALVO_NOME = "CHL Total"
ALVO_CHAVE = "chl_total"
MAPA = {
    "23_02": ("23/02", "CHL TOTAL (23/02)", "D02M"),
    "24_02": ("24/02", "CHL TOTAL (24/02)", "D03M"),
    "25_02": ("25/02", "CHL TOTAL (25.02)", "D04M"),
    "26_02": ("26/02", "CHL TOTAL (26/02)", "D05M"),
    "27_02": ("27/02", "CHL TOTAL (27/02)", "D06M"),
    "02_03": ("02/03", "CHL TOTAL (02/03)", "D09M"),
    "03_03": ("03/03", "CHL TOTAL (03/03)", "D10M"),
}
MAX_COMPONENTES = 5
LIMIAR_ABS_CORRELACAO = .80


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
    tab = pd.DataFrame(linhas)
    return int(tab.loc[tab.rmse_cv_bloco.idxmin(), "n_componentes"]), tab


def predicao_cv(X, y, grupos, n):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=len(np.unique(grupos))).split(X, y, grupos):
        pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
    return pred


def tabela_markdown(df, casas=4):
    cab = [str(c) for c in df.columns]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "|".join(["---"] * len(cab)) + "|"]
    for _, linha in df.iterrows():
        vals = [f"{v:.{casas}f}" if isinstance(v, (float, np.floating)) else str(v) for v in linha]
        linhas.append("| " + " | ".join(vals) + " |")
    return "\n".join(linhas)


def plotar(dados, y, pred, data, caminho):
    cores, rotulos = {"IRRIG": "#1976d2", "NIRRIG": "#d32f2f"}, {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
    fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    for condicao in ("IRRIG", "NIRRIG"):
        mask = dados.condicao.eq(condicao).to_numpy()
        ax.scatter(y[mask], pred[mask], s=62, color=cores[condicao], edgecolor="white", linewidth=.7,
                   label=rotulos[condicao], zorder=3)
    minimo, maximo = min(y.min(), pred.min()), max(y.max(), pred.max())
    margem = max((maximo - minimo) * .08, .2)
    ax.plot([minimo-margem, maximo+margem], [minimo-margem, maximo+margem], "--", color="#4a5568", lw=1.2, label="1:1")
    ax.text(.04, .96, f"R² CV = {r2_score(y, pred):.3f}\nRMSE CV = {mean_squared_error(y, pred)**.5:.3f}",
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#a0aec0", "boxstyle": "round,pad=.35"})
    ax.set(xlim=(minimo-margem, maximo+margem), ylim=(minimo-margem, maximo+margem),
           xlabel=f"{ALVO_NOME} observado", ylabel=f"{ALVO_NOME} predito (CV por bloco)",
           title=f"{ALVO_NOME} — {data}\nPLSR com as 5 bandas globais")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(caminho, dpi=240)
    plt.close(fig)


def carregar():
    fis = pd.read_excel(FISIO)
    fis["condicao"] = fis["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["genotipo"] = fis["Genótipo"]
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda i: f"B{i+1}")
    bruto = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    corrigido, w = preprocessar(bruto)
    bandas = w.astype(int); colunas = [str(b) for b in bandas]
    X = snv(savitzky_golay(corrigido[colunas].to_numpy(float), window_length=11, polyorder=2))
    meta_esp = corrigido[[c for c in corrigido.columns if not c.isdigit()]].reset_index(drop=True)
    esp = pd.concat([meta_esp, pd.DataFrame(X, columns=colunas)], axis=1)
    meta = ["bloco", "genotipo", "condicao"]
    partes = []
    for chave, (data, alvo, coleta) in MAPA.items():
        medias = (esp[(esp.data_coleta == coleta) & (esp.turno == "manha")]
                  .groupby(meta, as_index=False)[colunas].mean())
        parte = fis[meta + [alvo]].merge(medias, on=meta, how="inner", validate="one_to_one").dropna(subset=[alvo])
        parte = parte.rename(columns={alvo: "chl_total"})
        parte.insert(0, "chave_data", chave); parte.insert(1, "data_chl", data)
        partes.append(parte)
    return pd.concat(partes, ignore_index=True), bandas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dados, bandas = carregar(); colunas = [str(b) for b in bandas]
    X, y, grupos = dados[colunas].to_numpy(float), dados.chl_total.to_numpy(float), dados.bloco.to_numpy()
    corr_s = spearman_matriz(X); grupos_s = agrupar(corr_s, bandas, limiar_r=.80, janela_nm=10.0)
    rho_y = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = np.array([np.flatnonzero(grupos_s == g)[np.nanargmax(np.abs(rho_y[grupos_s == g]))] for g in np.unique(grupos_s)])
    Xr, br = X[:, reps], bandas[reps]
    n_rank, tab_rank = cv_componentes(Xr, y, grupos)
    modelo_rank = PLSRegression(n_components=n_rank, scale=True).fit(Xr, y)
    bruto = np.asarray(modelo_rank.coef_).reshape(-1)
    beta = bruto * modelo_rank._x_std / modelo_rank._y_std[0]
    ranking = pd.DataFrame({"indice_representante": np.arange(len(br)), "banda_nm": br,
                            "coeficiente_plsr_padronizado": beta, "coeficiente_plsr_absoluto": abs(beta),
                            "coeficiente_plsr_bruto": bruto, f"rho_spearman_{ALVO_CHAVE}": rho_y[reps]})
    ranking = ranking.sort_values("coeficiente_plsr_absoluto", ascending=False).reset_index(drop=True)
    ranking.insert(0, "ranking_coeficiente_plsr", np.arange(1, len(ranking)+1))
    corr_p = np.corrcoef(Xr, rowvar=False); escolhidas, max_corr = [], []
    for _, linha in ranking.iterrows():
        ind = int(linha.indice_representante)
        maior = max((abs(corr_p[ind, outro]) for outro in escolhidas), default=np.nan)
        max_corr.append(maior)
        if (not escolhidas or maior < LIMIAR_ABS_CORRELACAO) and len(escolhidas) < 5:
            escolhidas.append(ind)
    if len(escolhidas) < 5:
        raise ValueError("Não foi possível selecionar cinco bandas não colineares.")
    ranking["max_abs_r_pearson_com_banda_ja_selecionada"] = max_corr
    ranking["selecionada_top5_nao_colinear"] = ranking.indice_representante.isin(escolhidas)
    top5 = ranking[ranking.selecionada_top5_nao_colinear].sort_values("ranking_coeficiente_plsr").reset_index(drop=True)
    top5.insert(0, "ranking_top5_nao_colinear", np.arange(1, 6))
    top_bandas = top5.banda_nm.to_numpy(int); top_idx = np.array([np.where(bandas == b)[0][0] for b in top_bandas])
    ranking.to_csv(OUT / "ranking_global_coeficientes_plsr.csv", sep=";", index=False)
    top5.to_csv(OUT / "top5_bandas_globais_plsr.csv", sep=";", index=False)
    idx_rep = top5.indice_representante.to_numpy(int)
    pd.DataFrame(corr_p[np.ix_(idx_rep, idx_rep)], index=top_bandas, columns=top_bandas).rename_axis("banda_nm").to_csv(OUT / "correlacao_pearson_top5_nao_colinear.csv", sep=";")
    tab_rank.to_csv(OUT / "global_ranking_componentes_cv.csv", sep=";", index=False)
    Xtop = X[:, top_idx]; n_global, tab_global = cv_componentes(Xtop, y, grupos)
    mod_global = PLSRegression(n_components=n_global, scale=True).fit(Xtop, y)
    ajuste, cv = mod_global.predict(Xtop).ravel(), predicao_cv(Xtop, y, grupos, n_global)
    tab_global.to_csv(OUT / "global_top5_componentes_cv.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", "data_chl", "chl_total"]].assign(predito_ajuste=ajuste, predito_cv_bloco=cv, residuo_cv_bloco=y-cv).to_csv(OUT / "global_top5_predicoes.csv", sep=";", index=False)
    resumo = [{"escopo": "Global (todas as datas; Top 5)", "n": len(y), "componentes": n_global, "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": mean_squared_error(y, ajuste)**.5, "r2_cv_bloco": r2_score(y, cv), "rmse_cv_bloco": mean_squared_error(y, cv)**.5}]
    for chave, (data, _, coleta) in MAPA.items():
        dia = dados[dados.chave_data.eq(chave)].reset_index(drop=True)
        xd, yd, gd = dia[[str(b) for b in top_bandas]].to_numpy(float), dia.chl_total.to_numpy(float), dia.bloco.to_numpy()
        n, tab = cv_componentes(xd, yd, gd); mod = PLSRegression(n_components=n, scale=True).fit(xd, yd)
        ajuste_d, cv_d = mod.predict(xd).ravel(), predicao_cv(xd, yd, gd, n)
        tab.to_csv(OUT / f"{chave}_componentes_cv.csv", sep=";", index=False)
        dia[["bloco", "genotipo", "condicao", "data_chl", "chl_total"]].assign(predito_ajuste=ajuste_d, predito_cv_bloco=cv_d, residuo_cv_bloco=yd-cv_d).to_csv(OUT / f"{chave}_predicoes_top5.csv", sep=";", index=False)
        plotar(dia, yd, cv_d, data, OUT / f"{chave}_observado_vs_predito_cv_top5.png")
        resumo.append({"escopo": data, "coleta_espectral": coleta, "n": len(yd), "componentes": n, "r2_ajuste": r2_score(yd, ajuste_d), "rmse_ajuste": mean_squared_error(yd, ajuste_d)**.5, "r2_cv_bloco": r2_score(yd, cv_d), "rmse_cv_bloco": mean_squared_error(yd, cv_d)**.5})
    resumo = pd.DataFrame(resumo); resumo.to_csv(OUT / "resumo_metricas.csv", sep=";", index=False)
    texto = [f"# PLSR — {ALVO_NOME} global e predição por data", "", f"Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < {LIMIAR_ABS_CORRELACAO:.2f} com as já selecionadas.", "", "## Top 5 global", "", tabela_markdown(top5), "", "## Métricas", "", tabela_markdown(resumo), "", "As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada."]
    (OUT / "relatorio.md").write_text("\n".join(texto)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
