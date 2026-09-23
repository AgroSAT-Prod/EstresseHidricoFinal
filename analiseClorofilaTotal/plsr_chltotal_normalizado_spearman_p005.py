#!/usr/bin/env python3
"""PLSR de CHL Total: pré-processamento completo, Spearman p<0,05 e VIP.

Usa o mesmo fluxo do projeto: recorte 400--2450 nm, correção de emendas,
interpolação, Savitzky--Golay e SNV (estágio ``normalizado``). As sete datas
entram num único modelo. A redução local usa Spearman bilateral p < 0,05 e o
filtro final guloso impede pares de bandas com p < 0,05.
"""
from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_student
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "testeDeNormalidade"), str(ROOT / "reducaoColinearidade")])
from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar_por_pvalor, spearman_matriz  # noqa: E402

OUT = ROOT / "analiseClorofilaTotal" / "resultados" / "todos_dias_normalizado_spearman_p005_vip_nao_redundante"
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
MAPA = {
    "23/02": ("CHL TOTAL (23/02)", "D02M"), "24/02": ("CHL TOTAL (24/02)", "D03M"),
    "25/02": ("CHL TOTAL (25.02)", "D04M"), "26/02": ("CHL TOTAL (26/02)", "D05M"),
    "27/02": ("CHL TOTAL (27/02)", "D06M"), "02/03": ("CHL TOTAL (02/03)", "D09M"),
    "03/03": ("CHL TOTAL (03/03)", "D10M"),
}
P_LIMIAR, MAX_COMPONENTES = .05, 10


def calcular_vip(modelo):
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * ((wn ** 2) * ssy).sum(axis=1) / ssy.sum())


def p_de_rho(r, n):
    r = np.clip(np.abs(np.asarray(r)), 0, 1 - 1e-15)
    estat = r * np.sqrt((n - 2) / np.maximum(1 - r ** 2, 1e-30))
    return 2 * t_student.sf(estat, df=n - 2)


def escolher_componentes(X, y, grupos):
    folds = list(GroupKFold(n_splits=4).split(X, y, grupos))
    limite = min(MAX_COMPONENTES, X.shape[1], min(len(tr) - 1 for tr, _ in folds))
    linhas = []
    for n in range(1, limite + 1):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
        linhas.append({"componentes": n, "r2_cv_bloco": r2_score(y, pred), "rmse_cv_bloco": mean_squared_error(y, pred) ** .5})
    tabela = pd.DataFrame(linhas)
    return int(tabela.loc[tabela.rmse_cv_bloco.idxmin(), "componentes"]), tabela


def medias_bloco(meta, X, dia):
    chaves = ["bloco", "genotipo", "condicao"]
    submeta = meta.loc[meta.data_coleta.eq(dia), chaves].reset_index(drop=True)
    subx = X[meta.data_coleta.eq(dia).to_numpy()]
    indices = submeta.groupby(chaves, sort=True).indices
    linhas, medias = [], []
    for valores, idx in indices.items():
        linhas.append(dict(zip(chaves, valores)))
        medias.append(subx[idx].mean(axis=0))
    return pd.DataFrame(linhas), np.asarray(medias)


def filtrar(vip, reps, corr, n):
    """Greedy por VIP: rejeita toda candidata com p Spearman < 0,05."""
    mantidas = []
    for pos in np.argsort(vip)[::-1]:
        if not mantidas or np.all(p_de_rho(corr[reps[pos], reps[mantidas]], n) >= P_LIMIAR):
            mantidas.append(pos)
    return np.asarray(mantidas, dtype=int)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--excluir-dia", choices=[dia for _, dia in MAPA.values()])
    args = parser.parse_args()
    mapa = {data: valores for data, valores in MAPA.items() if valores[1] != args.excluir_dia}
    if args.excluir_dia:
        OUT = ROOT / "analiseClorofilaTotal" / "resultados" / f"todos_dias_normalizado_spearman_p005_vip_nao_redundante_sem_{args.excluir_dia[:-1]}"
    OUT.mkdir(parents=True, exist_ok=True)
    fis = pd.read_excel(FISIO)
    fis["genotipo"] = fis["Genótipo"]
    fis["condicao"] = fis["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda i: f"B{i + 1}")
    meta, espectro, bandas = carregar("normalizado", turno="manha")
    chaves = ["bloco", "genotipo", "condicao"]
    partes = []
    for data, (alvo, dia) in mapa.items():
        metad, xd = medias_bloco(meta, espectro, dia)
        espectro_df = pd.DataFrame(xd, columns=bandas.astype(int).astype(str))
        medias = pd.concat([metad, espectro_df], axis=1)
        d = fis[chaves + [alvo]].merge(medias, on=chaves, how="inner", validate="one_to_one").dropna(subset=[alvo]).rename(columns={alvo: "chl_total"})
        d.insert(0, "data", data)
        partes.append(d)
    dados = pd.concat(partes, ignore_index=True)
    y, grupos = dados.chl_total.to_numpy(float), dados.bloco.to_numpy()
    X = dados[bandas.astype(int).astype(str)].to_numpy(float)
    corr = spearman_matriz(X)
    grupos_rho = agrupar_por_pvalor(corr, bandas, len(y), p_limiar=P_LIMIAR, janela_nm=10.0)
    rho_y = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = np.array([idx[np.nanargmax(np.abs(rho_y[idx]))] for g in np.unique(grupos_rho) if (idx := np.flatnonzero(grupos_rho == g)).size])
    Xr, br = X[:, reps], bandas[reps]
    n_pre, _ = escolher_componentes(Xr, y, grupos)
    vip_pre = calcular_vip(PLSRegression(n_components=n_pre, scale=True).fit(Xr, y))
    cand = vip_pre >= 1
    if not cand.any(): cand[np.argmax(vip_pre)] = True
    Xc, bc, rc, vc = Xr[:, cand], br[cand], reps[cand], vip_pre[cand]
    n_cand, _ = escolher_componentes(Xc, y, grupos)
    modelo_cand = PLSRegression(n_components=n_cand, scale=True).fit(Xc, y)
    vip_cand = calcular_vip(modelo_cand)
    manter = filtrar(vip_cand, rc, corr, len(y))
    Xf, bf, rf = Xc[:, manter], bc[manter], rc[manter]
    n_final, tabela = escolher_componentes(Xf, y, grupos)
    modelo = PLSRegression(n_components=n_final, scale=True).fit(Xf, y)
    ajuste = modelo.predict(Xf).ravel()
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=4).split(Xf, y, grupos):
        pred[te] = PLSRegression(n_components=n_final, scale=True).fit(Xf[tr], y[tr]).predict(Xf[te]).ravel()
    ranking = pd.DataFrame({"banda_nm": bf.astype(int), "vip_criterio": vc[manter], "vip_antes_filtro": vip_cand[manter], "vip_modelo_final": calcular_vip(modelo), "rho_spearman_chl_total": rho_y[rf]}).sort_values("vip_modelo_final", ascending=False)
    ranking.to_csv(OUT / "bandas_finais.csv", sep=";", index=False)
    tabela.to_csv(OUT / "componentes_cv.csv", sep=";", index=False)
    dados[chaves + ["data", "chl_total"]].assign(predito_cv=pred, residuo_cv=y-pred).to_csv(OUT / "predicoes_cv_bloco.csv", sep=";", index=False)
    resumo = pd.DataFrame([{"amostras": len(y), "representantes_spearman_p005": len(br), "candidatas_vip_ge_1": int(cand.sum()), "bandas_apos_filtro": len(bf), "componentes": n_final, "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": mean_squared_error(y, ajuste) ** .5, "r2_cv_bloco": r2_score(y, pred), "rmse_cv_bloco": mean_squared_error(y, pred) ** .5}])
    resumo.to_csv(OUT / "resumo.csv", sep=";", index=False)
    r = resumo.iloc[0]
    texto = f"""# PLSR — CHL Total, {len(mapa)} datas, espectro normalizado

Pré-processamento: correção das emendas, limpeza/interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. As coletas {', '.join(mapa)} foram reunidas em um modelo ({int(r.amostras)} observações).{" A coleta " + args.excluir_dia + " foi excluída." if args.excluir_dia else ""}

Redução: Spearman bilateral **p < 0,05** em bandas contíguas, janela <10 nm. Após VIP ≥1, aplicou-se filtro guloso de não redundância: uma banda só permanece se sua correlação Spearman com todas as bandas VIP de maior importância tiver p ≥0,05. A validação é GroupKFold por bloco (quatro dobras).

| Métrica | Resultado |
|---|---:|
| Representantes Spearman p <0,05 | {int(r.representantes_spearman_p005)} |
| Candidatas VIP ≥1 | {int(r.candidatas_vip_ge_1)} |
| Bandas após filtro | {int(r.bandas_apos_filtro)} |
| Componentes PLSR | {int(r.componentes)} |
| R² ajuste | {r.r2_ajuste:.3f} |
| RMSE ajuste | {r.rmse_ajuste:.3f} |
| **R² CV por bloco** | **{r.r2_cv_bloco:.3f}** |
| **RMSE CV por bloco** | **{r.rmse_cv_bloco:.3f}** |
"""
    (OUT / "relatorio.md").write_text(texto, encoding="utf-8")
    plt.figure(figsize=(7.5, 6.5)); plt.scatter(y, pred, color="#147d91", edgecolor="white", alpha=.8)
    lo, hi = min(y.min(), pred.min())-.7, max(y.max(), pred.max())+.7
    plt.plot([lo, hi], [lo, hi], "--", color="#b43c2f", label="1:1"); plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.gca().set_aspect("equal"); plt.xlabel("CHL Total observado"); plt.ylabel("CHL Total predito (CV por bloco)")
    plt.title(f"PLSR — CHL Total normalizado, {len(mapa)} coletas"); plt.text(.04,.95,f"R² CV = {r.r2_cv_bloco:.3f}\nRMSE CV = {r.rmse_cv_bloco:.3f}",transform=plt.gca().transAxes,va="top",bbox={"facecolor":"white","edgecolor":"#aaa"}); plt.grid(alpha=.18); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "predicao_cv.png",dpi=300,bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
