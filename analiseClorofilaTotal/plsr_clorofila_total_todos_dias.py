#!/usr/bin/env python3
"""PLSR único de Clorofila Total usando conjuntamente as sete coletas."""
from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analiseYield"))
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))
import plsr_yield_tres_datas as base  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

OUT_BASE = ROOT / "analiseClorofilaTotal" / "resultados"
MAPA = {
    "23/02": ("CHL TOTAL (23/02)", "D02M"),
    "24/02": ("CHL TOTAL (24/02)", "D03M"),
    "25/02": ("CHL TOTAL (25.02)", "D04M"),
    "26/02": ("CHL TOTAL (26/02)", "D05M"),
    "27/02": ("CHL TOTAL (27/02)", "D06M"),
    "02/03": ("CHL TOTAL (02/03)", "D09M"),
    "03/03": ("CHL TOTAL (03/03)", "D10M"),
}
MAX_COMPONENTES = 10


def vip(modelo):
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * ((wn ** 2) * ssy).sum(axis=1) / ssy.sum())


def componentes_cv(X, y, grupos):
    folds = list(GroupKFold(n_splits=4).split(X, y, grupos))
    limite = min(MAX_COMPONENTES, X.shape[1], min(len(tr) - 1 for tr, _ in folds))
    linhas = []
    for n in range(1, limite + 1):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
        linhas.append({"componentes": n, "r2_cv_bloco": r2_score(y, pred),
                       "rmse_cv_bloco": mean_squared_error(y, pred) ** .5})
    tab = pd.DataFrame(linhas)
    return int(tab.loc[tab.rmse_cv_bloco.idxmin(), "componentes"]), tab


def filtrar_nao_redundantes(vip_atual, reps, corr, limiar=.80):
    """Mantém VIPs em ordem decrescente sem pares com |rho| >= limiar."""
    ordem = np.argsort(vip_atual)[::-1]
    mantidas = []
    for pos in ordem:
        if not mantidas or np.all(np.abs(corr[reps[pos], reps[mantidas]]) < limiar):
            mantidas.append(pos)
    return np.array(mantidas, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nao-redundante", action="store_true",
                        help="Aplica filtro final guloso |rho| < 0,80 entre VIPs.")
    args = parser.parse_args()
    out = OUT_BASE / ("todos_dias_vip_nao_redundante" if args.nao_redundante else "todos_dias")
    out.mkdir(parents=True, exist_ok=True)
    fis, esp, bandas = base.preparar()
    meta = ["bloco", "genotipo", "condicao"]
    partes = []
    for data, (alvo, dia) in MAPA.items():
        sub = esp[(esp.data_coleta == dia) & (esp.turno == "manha")]
        medias = sub.groupby(meta, as_index=False)[[str(b) for b in bandas]].mean()
        d = fis[meta + [alvo]].merge(medias, on=meta, how="inner", validate="one_to_one")
        d = d.dropna(subset=[alvo]).rename(columns={alvo: "chl_total"})
        d.insert(0, "data", data)
        partes.append(d)
    dados = pd.concat(partes, ignore_index=True)
    y = dados.chl_total.to_numpy(float)
    X = dados[[str(b) for b in bandas]].to_numpy(float)
    grupos = dados.bloco.to_numpy()

    corr = spearman_matriz(X)
    grupos_rho = agrupar(corr, bandas, limiar_r=.80, janela_nm=10.0)
    rho_y = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = []
    for grupo in np.unique(grupos_rho):
        idx = np.flatnonzero(grupos_rho == grupo)
        reps.append(idx[np.nanargmax(np.abs(rho_y[idx]))])
    reps = np.array(reps)
    Xr, br = X[:, reps], bandas[reps]
    n_pre, _ = componentes_cv(Xr, y, grupos)
    modelo_pre = PLSRegression(n_components=n_pre, scale=True).fit(Xr, y)
    vip_pre = vip(modelo_pre)
    seleciona = vip_pre >= 1
    if not seleciona.any():
        seleciona[np.argmax(vip_pre)] = True
    Xs, bs, reps_sel, vip_criterio = Xr[:, seleciona], br[seleciona], reps[seleciona], vip_pre[seleciona]
    n_final, tab = componentes_cv(Xs, y, grupos)
    modelo = PLSRegression(n_components=n_final, scale=True).fit(Xs, y)
    vip_antes_filtro = vip(modelo)
    if args.nao_redundante:
        mantidas = filtrar_nao_redundantes(vip_antes_filtro, reps_sel, corr)
        Xs, bs, reps_sel = Xs[:, mantidas], bs[mantidas], reps_sel[mantidas]
        vip_criterio = vip_criterio[mantidas]
        vip_selecao_final = vip_antes_filtro[mantidas]
        n_final, tab = componentes_cv(Xs, y, grupos)
        modelo = PLSRegression(n_components=n_final, scale=True).fit(Xs, y)
    else:
        vip_selecao_final = vip_antes_filtro
    ajustado = modelo.predict(Xs).ravel()
    predito = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=4).split(Xs, y, grupos):
        predito[te] = PLSRegression(n_components=n_final, scale=True).fit(Xs[tr], y[tr]).predict(Xs[te]).ravel()
    ranking = pd.DataFrame({"banda_nm": bs, "vip_criterio_selecao": vip_criterio,
                            "vip_antes_filtro": vip_selecao_final,
                            "vip_modelo_final": vip(modelo), "rho_spearman_chl_total": rho_y[reps_sel]}).sort_values("vip_modelo_final", ascending=False)
    ranking.to_csv(out / "bandas_vip_ge_1.csv", sep=";", index=False)
    tab.to_csv(out / "componentes_cv.csv", sep=";", index=False)
    dados[meta + ["data", "chl_total"]].assign(predito_cv=predito, residuo_cv=y-predito).to_csv(out / "predicoes_cv_bloco.csv", sep=";", index=False)
    resumo = pd.DataFrame([{
        "amostras": len(y), "datas": len(MAPA), "representantes_spearman": len(br),
        "bandas_apos_filtro": len(bs), "componentes": n_final,
        "r2_ajuste": r2_score(y, ajustado), "rmse_ajuste": mean_squared_error(y, ajustado) ** .5,
        "r2_cv_bloco": r2_score(y, predito), "rmse_cv_bloco": mean_squared_error(y, predito) ** .5,
    }])
    resumo.to_csv(out / "resumo.csv", sep=";", index=False)
    r = resumo.iloc[0]
    texto = f"""# PLSR — Clorofila Total, todas as sete coletas

As sete datas foram combinadas num único modelo ({len(y)} observações; D02–D06, D09 e D10). As leituras foram agregadas por bloco × genótipo × condição. Aplicou-se Spearman |ρ| ≥ 0,80 entre bandas contíguas, selecionando uma representante por grupo pela maior associação absoluta com CHL Total; o PLSR escalonado reteve VIP ≥ 1,0.{" Ao final, aplicou-se filtro guloso: uma banda só permanece se tiver |ρ| < 0,80 com todas as bandas VIP de maior importância já mantidas." if args.nao_redundante else ""}

A validação foi GroupKFold com quatro dobras: cada dobra exclui um bloco inteiro, em todas as datas. O número de componentes foi escolhido pelo menor RMSE dessa CV.

| Métrica | Resultado |
|---|---:|
| Amostras válidas | {int(r.amostras)} |
| Representantes após Spearman | {int(r.representantes_spearman)} |
| Bandas {"após filtro de não redundância" if args.nao_redundante else "VIP ≥ 1"} | {int(r.bandas_apos_filtro)} |
| Componentes PLSR | {int(r.componentes)} |
| R² de ajuste | {r.r2_ajuste:.3f} |
| RMSE de ajuste | {r.rmse_ajuste:.3f} |
| **R² CV por bloco** | **{r.r2_cv_bloco:.3f}** |
| **RMSE CV por bloco** | **{r.rmse_cv_bloco:.3f}** |

O R²/RMSE CV é diagnóstico porque a seleção Spearman/VIP foi feita antes das dobras; uma estimativa estritamente imparcial exigiria seleção aninhada em cada dobra.
"""
    (out / "relatorio.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
