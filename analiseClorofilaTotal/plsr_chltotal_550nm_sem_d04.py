#!/usr/bin/env python3
"""PLSR univariado de CHL Total com a banda 550 nm, sem D04M."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

OUT = ROOT / "analiseClorofilaTotal" / "resultados" / "somente_550nm_normalizado_sem_D04"
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
MAPA = {
    "23/02": ("CHL TOTAL (23/02)", "D02M"), "24/02": ("CHL TOTAL (24/02)", "D03M"),
    "26/02": ("CHL TOTAL (26/02)", "D05M"), "27/02": ("CHL TOTAL (27/02)", "D06M"),
    "02/03": ("CHL TOTAL (02/03)", "D09M"), "03/03": ("CHL TOTAL (03/03)", "D10M"),
}
BANDA = 550


def medias_bloco(meta, X, dia):
    chaves = ["bloco", "genotipo", "condicao"]
    mask = meta.data_coleta.eq(dia).to_numpy()
    m = meta.loc[mask, chaves].reset_index(drop=True)
    grupos = m.groupby(chaves, sort=True).indices
    linhas, valores = [], []
    for chave, idx in grupos.items():
        linhas.append(dict(zip(chaves, chave)))
        valores.append(X[mask][idx].mean(axis=0))
    return pd.DataFrame(linhas), np.asarray(valores)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fis = pd.read_excel(FISIO)
    fis["genotipo"] = fis["Genótipo"]
    fis["condicao"] = fis["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda i: f"B{i + 1}")
    meta, espectro, bandas = carregar("normalizado", turno="manha")
    idx = int(np.where(bandas.astype(int) == BANDA)[0][0])
    chaves, partes = ["bloco", "genotipo", "condicao"], []
    for data, (alvo, dia) in MAPA.items():
        md, xm = medias_bloco(meta, espectro[:, [idx]], dia)
        d = fis[chaves + [alvo]].merge(md.assign(**{str(BANDA): xm[:, 0]}), on=chaves, validate="one_to_one")
        d = d.dropna(subset=[alvo]).rename(columns={alvo: "chl_total"})
        d.insert(0, "data", data); partes.append(d)
    dados = pd.concat(partes, ignore_index=True)
    X, y, grupos = dados[[str(BANDA)]].to_numpy(float), dados.chl_total.to_numpy(float), dados.bloco.to_numpy()
    modelo = PLSRegression(n_components=1, scale=True).fit(X, y)
    ajuste = modelo.predict(X).ravel()
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=4).split(X, y, grupos):
        pred[te] = PLSRegression(n_components=1, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
    resumo = pd.DataFrame([{"banda_nm": BANDA, "amostras": len(y), "componentes": 1, "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": mean_squared_error(y, ajuste) ** .5, "r2_cv_bloco": r2_score(y, pred), "rmse_cv_bloco": mean_squared_error(y, pred) ** .5}])
    resumo.to_csv(OUT / "resumo.csv", sep=";", index=False)
    dados[chaves + ["data", "chl_total"]].assign(predito_cv=pred, residuo_cv=y-pred).to_csv(OUT / "predicoes_cv_bloco.csv", sep=";", index=False)
    r = resumo.iloc[0]
    (OUT / "relatorio.md").write_text(f"""# PLSR — CHL Total com 550 nm, sem D04M

Pré-processamento: recorte, correção das emendas, interpolação, Savitzky–Golay e SNV. Entram seis datas, excluindo D04M (25/02). Modelo univariado com somente 550 nm; validação GroupKFold por bloco (k=4).

| Métrica | Resultado |
|---|---:|
| Amostras | {int(r.amostras)} |
| Banda | 550 nm |
| R² ajuste | {r.r2_ajuste:.3f} |
| RMSE ajuste | {r.rmse_ajuste:.3f} |
| **R² CV por bloco** | **{r.r2_cv_bloco:.3f}** |
| **RMSE CV por bloco** | **{r.rmse_cv_bloco:.3f}** |
""", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(y, pred, color="#147d91", edgecolor="white", alpha=.85)
    lo, hi = min(y.min(), pred.min())-.7, max(y.max(), pred.max())+.7
    ax.plot([lo, hi], [lo, hi], "--", color="#b43c2f", label="1:1"); ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect="equal", xlabel="CHL Total observado", ylabel="CHL Total predito (CV por bloco)", title="PLSR univariado — 550 nm, sem D04M")
    ax.text(.04,.95,f"R² CV = {r.r2_cv_bloco:.3f}\nRMSE CV = {r.rmse_cv_bloco:.3f}",transform=ax.transAxes,va="top",bbox={"facecolor":"white","edgecolor":"#aaa"}); ax.grid(alpha=.18); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "predicao_cv.png",dpi=300,bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
