#!/usr/bin/env python3
"""PLSR de CRA agrupando 24/02, 02/03 e 03/03 por bloco (n=71)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import clorofila_total_spearman_plsr as base
import cra_interval_pls_vip as ipls

ROOT=Path(__file__).resolve().parents[1]
SAIDA=Path(__file__).resolve().parent/"resultados_cra_todos_dias_n71_por_bloco"
BANDAS=[str(i) for i in range(350,2501)]
DATAS={"24/02":"D03M","02/03":"D09M","03/03":"D10M"}
ESTILOS={"IRR":{"marker":"s","color":"#2563eb","label":"Irrigado"},"NIR":{"marker":"^","color":"#dc2626","label":"Não irrigado"}}

def prever(x,y,n,scale): return cross_val_predict(PLSRegression(n_components=n,scale=scale),x,y,cv=LeaveOneOut()).ravel()

def carregar():
    fis=pd.read_excel(ROOT/"dataset"/"FisiologicosEduarda.xlsx").rename(columns=lambda c:str(c).replace(".","/"))
    fis=fis.rename(columns={"Genótipo":"genotipo","Condição":"condicao"});fis["condicao"]=fis["condicao"].map({"IRR":"IRR","NIR":"NIR"})
    partes=[]
    for data,dia in DATAS.items():
        alvo=f"CRA ({data})"; parte=fis[["genotipo","condicao",alvo]].rename(columns={alvo:"cra"}).copy()
        parte["data_fisiologica"]=data;parte["cra"]=pd.to_numeric(parte["cra"],errors="coerce")
        # A posição é preservada antes da retirada da única CRA ausente (BR16/IRR/B1 em 02/03).
        parte["ordem_repeticao_fisiologia"]=parte.groupby(["genotipo","condicao"],sort=False).cumcount()+1
        parte["bloco"]="B"+parte["ordem_repeticao_fisiologia"].astype(str);partes.append(parte.dropna(subset=["cra"]))
    fisiologia=pd.concat(partes,ignore_index=True)
    esp=pd.read_csv(ROOT/"dataset"/"Unificada13052026_Limpa.csv",sep=";")
    esp=esp.loc[esp.data_coleta.isin(DATAS.values())&esp.turno.eq("manha")].copy();esp["data_fisiologica"]=esp.data_coleta.map({v:k for k,v in DATAS.items()});esp["condicao"]=esp.condicao.map({"IRRIG":"IRR","NIRRIG":"NIR"})
    for b in BANDAS: esp[b]=pd.to_numeric(esp[b].astype(str).str.replace(",",".",regex=False),errors="coerce")
    esp=esp.groupby(["data_fisiologica","bloco","genotipo","condicao"],as_index=False)[BANDAS].mean()
    dados=fisiologia.merge(esp,on=["data_fisiologica","bloco","genotipo","condicao"],validate="one_to_one")
    if len(dados)!=71: raise ValueError(f"Esperavam-se 71 pares, mas foram obtidos {len(dados)}.")
    return dados

def plotar(dados,y,ajuste,loo,r2,rmse):
    fig,eixos=plt.subplots(1,2,figsize=(10.5,4.6),layout="constrained")
    for eixo,pred,titulo in ((eixos[0],ajuste,"Ajuste"),(eixos[1],loo,"Validação leave-one-out")):
        for cond,estilo in ESTILOS.items():
            m=dados.condicao.eq(cond);eixo.scatter(y[m],pred[m],s=42,edgecolor="white",linewidth=.5,**estilo)
        baixo,alto=min(y.min(),pred.min())-.03,max(y.max(),pred.max())+.03;eixo.plot([baixo,alto],[baixo,alto],"--",color="#475569",linewidth=1)
        eixo.set(xlim=(baixo,alto),ylim=(baixo,alto),xlabel="CRA observado",ylabel="CRA predito",title=titulo)
    eixos[0].legend(frameon=False);eixos[1].text(.04,.95,f"n = 71\nR² LOO = {r2:.3f}\nRMSE LOO = {rmse:.3f}",transform=eixos[1].transAxes,va="top",bbox={"facecolor":"white","edgecolor":"#94a3b8","boxstyle":"round,pad=.3"})
    fig.suptitle("CRA — 24/02, 02/03 e 03/03: PLSR por bloco (n = 71)");fig.savefig(SAIDA/"observado_vs_predito_n71.png",dpi=220);plt.close(fig)

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING);SAIDA.mkdir(exist_ok=True);dados=carregar();y=dados.cra.to_numpy(float);x=dados[BANDAS].to_numpy(float)
    corr=[base.spearman_com_p_exato(x[:,i],y) for i in range(x.shape[1])];tab=pd.DataFrame({"banda_nm":np.asarray(BANDAS,dtype=int),"rho_spearman":[v[0] for v in corr],"p_valor":[v[1] for v in corr]})
    retidas=tab.loc[tab.p_valor<.05,"banda_nm"].astype(str).tolist();escolhidas=ipls.selecionar_ipls_diverso(dados,retidas,y,tab,largura_intervalo_nm=10,limiar_correlacao=None,n_bandas=5).reset_index(drop=True)
    bandas=escolhidas.banda_nm.astype(int).tolist();xf=dados[[str(b) for b in bandas]].to_numpy(float);espaco={"n_componentes":[1,2,3,4,5],"scale":[True,False]}
    def objetivo(trial):
        pred=prever(xf,y,int(trial.suggest_categorical("n_componentes",espaco["n_componentes"])),bool(trial.suggest_categorical("scale",espaco["scale"])));trial.set_user_attr("r2_loo",r2_score(y,pred));return root_mean_squared_error(y,pred)
    estudo=optuna.create_study(direction="minimize",sampler=optuna.samplers.GridSampler(espaco,seed=20260928));estudo.optimize(objetivo,n_trials=10,show_progress_bar=False)
    n,scale=int(estudo.best_params["n_componentes"]),bool(estudo.best_params["scale"]);loo=prever(xf,y,n,scale);modelo=PLSRegression(n_components=n,scale=scale).fit(xf,y);ajuste=modelo.predict(xf).ravel();r2,rmse=r2_score(y,loo),root_mean_squared_error(y,loo)
    escolhidas["vip_plsr_final"]=base.calcular_vip(modelo);escolhidas.sort_values("vip_plsr_final",ascending=False).to_csv(SAIDA/"top5_vip.csv",sep=";",index=False)
    dados[["data_fisiologica","bloco","genotipo","condicao","ordem_repeticao_fisiologia","cra"]].assign(predito_ajuste=ajuste,predito_loo=loo).to_csv(SAIDA/"dados_pareados_n71_predicoes.csv",sep=";",index=False)
    estudo.trials_dataframe(attrs=("number","value","params","state","user_attrs")).to_csv(SAIDA/"trials_optuna.csv",sep=";",index=False)
    pd.DataFrame([{"n":len(dados),"datas":"24/02, 02/03, 03/03","p_limiar_spearman":.05,"n_bandas_spearman":len(retidas),"bandas_nm":", ".join(map(str,bandas)),"n_componentes":n,"scale":scale,"r2_ajuste":r2_score(y,ajuste),"rmse_ajuste":root_mean_squared_error(y,ajuste),"r2_loo":r2,"rmse_loo":rmse,"pareamento":"ordem das repetições fisiológicas mapeada para B1-B4; CRA de BR16/IRR/B1 ausente em 02/03"}]).to_csv(SAIDA/"resumo_n71.csv",sep=";",index=False)
    plotar(dados,y,ajuste,loo,r2,rmse)
if __name__=="__main__":main()
