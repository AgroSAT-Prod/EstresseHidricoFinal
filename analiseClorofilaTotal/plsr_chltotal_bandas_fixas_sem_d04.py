#!/usr/bin/env python3
"""PLSR de CHL Total com bandas pré-definidas, sem D04M."""
from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

OUT = ROOT / "analiseClorofilaTotal" / "resultados" / "bandas_fixas_400_531_570_739_1580_normalizado_sem_D04"
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
BANDAS = np.array([400, 531, 570, 739, 1580])
USAR_GROUP_KFOLD = True
MAPA = {
    "23/02": ("CHL TOTAL (23/02)", "D02M"), "24/02": ("CHL TOTAL (24/02)", "D03M"),
    "26/02": ("CHL TOTAL (26/02)", "D05M"), "27/02": ("CHL TOTAL (27/02)", "D06M"),
    "02/03": ("CHL TOTAL (02/03)", "D09M"), "03/03": ("CHL TOTAL (03/03)", "D10M"),
}


def medias_bloco(meta, X, dia):
    chaves, mask = ["bloco", "genotipo", "condicao"], meta.data_coleta.eq(dia).to_numpy()
    m, subx = meta.loc[mask, chaves].reset_index(drop=True), X[mask]
    linhas, valores = [], []
    for chave, idx in m.groupby(chaves, sort=True).indices.items():
        linhas.append(dict(zip(chaves, chave))); valores.append(subx[idx].mean(axis=0))
    return pd.DataFrame(linhas), np.asarray(valores)


def componentes_cv(X, y, grupos):
    cv = GroupKFold(n_splits=4) if USAR_GROUP_KFOLD else KFold(n_splits=4, shuffle=True, random_state=42)
    folds = list(cv.split(X, y, grupos) if USAR_GROUP_KFOLD else cv.split(X, y))
    resultado = []
    for n in range(1, min(len(BANDAS), min(len(tr)-1 for tr, _ in folds)) + 1):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr],y[tr]).predict(X[te]).ravel()
        resultado.append({"componentes":n,"r2_cv_bloco":r2_score(y,pred),"rmse_cv_bloco":mean_squared_error(y,pred)**.5})
    d = pd.DataFrame(resultado)
    return int(d.loc[d.rmse_cv_bloco.idxmin(),"componentes"]), d


def main():
    global BANDAS, OUT, USAR_GROUP_KFOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandas", nargs="+", type=int, default=BANDAS.tolist())
    parser.add_argument("--kfold", action="store_true", help="Usa KFold=4 aleatório, sem grupos de bloco.")
    args = parser.parse_args()
    BANDAS = np.array(args.bandas)
    USAR_GROUP_KFOLD = not args.kfold
    if BANDAS.tolist() != [400, 531, 570, 739, 1580]:
        OUT = ROOT / "analiseClorofilaTotal" / "resultados" / ("bandas_fixas_" + "_".join(map(str, BANDAS)) + "_normalizado_sem_D04")
    if args.kfold:
        OUT = Path(str(OUT) + "_kfold4")
    OUT.mkdir(parents=True, exist_ok=True)
    fis = pd.read_excel(FISIO); fis["genotipo"] = fis["Genótipo"]; fis["condicao"] = fis["Condição"].replace({"IRR":"IRRIG","NIR":"NIRRIG"})
    fis["bloco"] = fis.groupby(["genotipo","condicao"],sort=False).cumcount().map(lambda i:f"B{i+1}")
    meta, espectro, ondas = carregar("normalizado", turno="manha")
    indices = np.array([np.where(ondas.astype(int)==b)[0][0] for b in BANDAS])
    chaves, partes = ["bloco","genotipo","condicao"], []
    for data,(alvo,dia) in MAPA.items():
        md,xm = medias_bloco(meta,espectro[:,indices],dia)
        d=fis[chaves+[alvo]].merge(pd.concat([md,pd.DataFrame(xm,columns=BANDAS.astype(str))],axis=1),on=chaves,validate="one_to_one")
        d=d.dropna(subset=[alvo]).rename(columns={alvo:"chl_total"});d.insert(0,"data",data);partes.append(d)
    dados=pd.concat(partes,ignore_index=True);X=dados[BANDAS.astype(str)].to_numpy(float);y=dados.chl_total.to_numpy(float);grupos=dados.bloco.to_numpy()
    n, tab=componentes_cv(X,y,grupos); modelo=PLSRegression(n_components=n,scale=True).fit(X,y);ajuste=modelo.predict(X).ravel();pred=np.full(len(y),np.nan)
    cv_final = GroupKFold(n_splits=4) if USAR_GROUP_KFOLD else KFold(n_splits=4, shuffle=True, random_state=42)
    divisoes = cv_final.split(X,y,grupos) if USAR_GROUP_KFOLD else cv_final.split(X,y)
    for tr,te in divisoes: pred[te]=PLSRegression(n_components=n,scale=True).fit(X[tr],y[tr]).predict(X[te]).ravel()
    r={"amostras":len(y),"bandas_nm":", ".join(map(str,BANDAS)),"componentes":n,"r2_ajuste":r2_score(y,ajuste),"rmse_ajuste":mean_squared_error(y,ajuste)**.5,"r2_cv_bloco":r2_score(y,pred),"rmse_cv_bloco":mean_squared_error(y,pred)**.5}
    pd.DataFrame([r]).to_csv(OUT/"resumo.csv",sep=";",index=False);tab.to_csv(OUT/"componentes_cv.csv",sep=";",index=False);dados[chaves+["data","chl_total"]].assign(predito_cv=pred,residuo_cv=y-pred).to_csv(OUT/"predicoes_cv_bloco.csv",sep=";",index=False)
    (OUT/"relatorio.md").write_text(f"""# PLSR — CHL Total com bandas fixas, sem D04M

Pré-processamento: correção das emendas, interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. Seis datas combinadas, excluindo D04M; {"GroupKFold por bloco" if USAR_GROUP_KFOLD else "KFold aleatório, sem agrupamento por bloco"} (k=4).

| Métrica | Resultado |
|---|---:|
| Amostras | {r['amostras']} |
| Bandas | {r['bandas_nm']} nm |
| Componentes PLSR | {n} |
| R² ajuste | {r['r2_ajuste']:.3f} |
| RMSE ajuste | {r['rmse_ajuste']:.3f} |
| **R² CV** | **{r['r2_cv_bloco']:.3f}** |
| **RMSE CV** | **{r['rmse_cv_bloco']:.3f}** |
""",encoding="utf-8")
    rotulo_cv = "CV por bloco" if USAR_GROUP_KFOLD else "KFold=4"
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(y,pred,color="#147d91",edgecolor="white",alpha=.85);lo,hi=min(y.min(),pred.min())-.7,max(y.max(),pred.max())+.7;ax.plot([lo,hi],[lo,hi],"--",color="#b43c2f",label="1:1");ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect="equal",xlabel="CHL Total observado",ylabel=f"CHL Total predito ({rotulo_cv})",title=f"PLSR — bandas {', '.join(map(str, BANDAS))} nm");ax.text(.04,.95,f"R² CV = {r['r2_cv_bloco']:.3f}\nRMSE CV = {r['rmse_cv_bloco']:.3f}",transform=ax.transAxes,va="top",bbox={"facecolor":"white","edgecolor":"#aaa"});ax.grid(alpha=.18);ax.legend();fig.tight_layout();fig.savefig(OUT/"predicao_cv.png",dpi=300,bbox_inches="tight");plt.close(fig)


if __name__ == "__main__": main()
