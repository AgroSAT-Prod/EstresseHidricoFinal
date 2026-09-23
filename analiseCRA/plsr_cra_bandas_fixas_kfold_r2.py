#!/usr/bin/env python3
"""PLSR de CRA com bandas fixas; KFold=4 e componentes por maior R² CV."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

OUT = ROOT / "analiseCRA" / "resultados" / "bandas_fixas_400_531_570_739_normalizado_kfold4_r2"
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
BANDAS = np.array([400, 531, 570, 739])
MAPA = {"24/02": ("CRA (24/02)", "D03M"), "02/03": ("CRA (02/03)", "D09M"), "03/03": ("CRA (03/03)", "D10M")}


def medias_bloco(meta, X, dia):
    chaves, mask = ["bloco", "genotipo", "condicao"], meta.data_coleta.eq(dia).to_numpy()
    m, subx = meta.loc[mask, chaves].reset_index(drop=True), X[mask]
    linhas, valores = [], []
    for chave, idx in m.groupby(chaves, sort=True).indices.items():
        linhas.append(dict(zip(chaves, chave))); valores.append(subx[idx].mean(axis=0))
    return pd.DataFrame(linhas), np.asarray(valores)


def selecionar_componentes(X, y):
    folds = list(KFold(n_splits=4, shuffle=True, random_state=42).split(X, y))
    linhas = []
    for n in range(1, min(X.shape[1], min(len(tr)-1 for tr, _ in folds)) + 1):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
        linhas.append({"componentes": n, "r2_cv": r2_score(y, pred), "rmse_cv": mean_squared_error(y, pred)**.5})
    tab = pd.DataFrame(linhas)
    return int(tab.loc[tab.r2_cv.idxmax(), "componentes"]), tab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fis = pd.read_excel(FISIO); fis["genotipo"] = fis["Genótipo"]; fis["condicao"] = fis["Condição"].replace({"IRR":"IRRIG","NIR":"NIRRIG"})
    fis["bloco"] = fis.groupby(["genotipo","condicao"],sort=False).cumcount().map(lambda i:f"B{i+1}")
    meta, esp, ondas = carregar("normalizado", turno="manha")
    idx = np.array([np.where(ondas.astype(int)==b)[0][0] for b in BANDAS]); chaves, partes = ["bloco","genotipo","condicao"], []
    for data,(alvo,dia) in MAPA.items():
        md,xm = medias_bloco(meta,esp[:,idx],dia)
        d=fis[chaves+[alvo]].merge(pd.concat([md,pd.DataFrame(xm,columns=BANDAS.astype(str))],axis=1),on=chaves,validate="one_to_one")
        d=d.dropna(subset=[alvo]).rename(columns={alvo:"cra"});d.insert(0,"data",data);partes.append(d)
    dados=pd.concat(partes,ignore_index=True);X=dados[BANDAS.astype(str)].to_numpy(float);y=dados.cra.to_numpy(float)
    n, tab=selecionar_componentes(X,y);modelo=PLSRegression(n_components=n,scale=True).fit(X,y);ajuste=modelo.predict(X).ravel();pred=np.full(len(y),np.nan)
    for tr,te in KFold(n_splits=4,shuffle=True,random_state=42).split(X,y): pred[te]=PLSRegression(n_components=n,scale=True).fit(X[tr],y[tr]).predict(X[te]).ravel()
    r={"amostras":len(y),"bandas_nm":", ".join(map(str,BANDAS)),"componentes":n,"r2_ajuste":r2_score(y,ajuste),"rmse_ajuste":mean_squared_error(y,ajuste)**.5,"r2_cv":r2_score(y,pred),"rmse_cv":mean_squared_error(y,pred)**.5}
    pd.DataFrame([r]).to_csv(OUT/"resumo.csv",sep=";",index=False);tab.to_csv(OUT/"componentes_kfold.csv",sep=";",index=False);dados[chaves+["data","cra"]].assign(predito_cv=pred,residuo_cv=y-pred).to_csv(OUT/"predicoes_kfold.csv",sep=";",index=False)
    (OUT/"relatorio.md").write_text(f"""# PLSR — CRA com bandas fixas

Pré-processamento: correção das emendas, interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. As três datas de CRA foram combinadas; a observação sem CRA em 02/03 foi excluída. Validação KFold=4 aleatório, `shuffle=True`, semente 42. O número de componentes foi escolhido pelo maior R² CV.

| Métrica | Resultado |
|---|---:|
| Amostras | {r['amostras']} |
| Bandas | {r['bandas_nm']} nm |
| Componentes (maior R² CV) | {n} |
| R² ajuste | {r['r2_ajuste']:.3f} |
| RMSE ajuste | {r['rmse_ajuste']:.3f} |
| **R² KFold=4** | **{r['r2_cv']:.3f}** |
| **RMSE KFold=4** | **{r['rmse_cv']:.3f}** |
""",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(y,pred,color="#147d91",edgecolor="white",alpha=.85);lo,hi=min(y.min(),pred.min())-2,max(y.max(),pred.max())+2;ax.plot([lo,hi],[lo,hi],"--",color="#b43c2f",label="1:1");ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect="equal",xlabel="CRA observado",ylabel="CRA predito (KFold=4)",title="PLSR — CRA: 400, 531, 570 e 739 nm");ax.text(.04,.95,f"R² CV = {r['r2_cv']:.3f}\nRMSE CV = {r['rmse_cv']:.3f}",transform=ax.transAxes,va="top",bbox={"facecolor":"white","edgecolor":"#aaa"});ax.grid(alpha=.18);ax.legend();fig.tight_layout();fig.savefig(OUT/"predicao_kfold.png",dpi=300,bbox_inches="tight");plt.close(fig)


if __name__ == "__main__": main()
