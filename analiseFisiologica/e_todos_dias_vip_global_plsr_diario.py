#!/usr/bin/env python3
"""Seleciona VIP global para E e ajusta PLSR diário com as cinco bandas fixas."""
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
SAIDA=Path(__file__).resolve().parent/"resultados_e_todos_dias_vip_global_plsr_diario"
BANDAS=[str(i) for i in range(350,2501)]
DATAS={"23/02":"D02M","24/02":"D03M","25/02":"D04M","26/02":"D05M","27/02":"D06M","02/03":"D09M","03/03":"D10M"}
ESTILOS={"IRR":{"marker":"s","color":"#2563eb","label":"Irrigado"},"NIR":{"marker":"^","color":"#dc2626","label":"Não irrigado"}}

def prever(x,y,n,scale): return cross_val_predict(PLSRegression(n_components=n,scale=scale),x,y,cv=LeaveOneOut()).ravel()

def carregar():
    fis=pd.read_excel(ROOT/"dataset"/"FisiologicosEduarda.xlsx").rename(columns=lambda c:str(c).replace(".","/"))
    fis=fis.rename(columns={"Genótipo":"genotipo","Condição":"condicao"});fis["condicao"]=fis.condicao.map({"IRR":"IRR","NIR":"NIR"})
    partes=[]
    for data in DATAS:
        alvo=f"E ({data})";parte=fis[["genotipo","condicao",alvo]].rename(columns={alvo:"e"}).copy();parte["data_fisiologica"]=data;parte["e"]=pd.to_numeric(parte["e"],errors="coerce")
        parte["ordem_repeticao_fisiologia"]=parte.groupby(["genotipo","condicao"],sort=False).cumcount()+1;parte["bloco"]="B"+parte.ordem_repeticao_fisiologia.astype(str);partes.append(parte.dropna(subset=["e"]))
    fis=pd.concat(partes,ignore_index=True)
    esp=pd.read_csv(ROOT/"dataset"/"Unificada13052026_Limpa.csv",sep=";");esp=esp.loc[esp.data_coleta.isin(DATAS.values())&esp.turno.eq("manha")].copy()
    esp["data_fisiologica"]=esp.data_coleta.map({v:k for k,v in DATAS.items()});esp["condicao"]=esp.condicao.map({"IRRIG":"IRR","NIRRIG":"NIR"})
    for b in BANDAS:esp[b]=pd.to_numeric(esp[b].astype(str).str.replace(",",".",regex=False),errors="coerce")
    esp=esp.groupby(["data_fisiologica","bloco","genotipo","condicao"],as_index=False)[BANDAS].mean()
    dados=fis.merge(esp,on=["data_fisiologica","bloco","genotipo","condicao"],validate="one_to_one")
    if len(dados)!=168:raise ValueError(f"Esperavam-se 168 pares; obtidos {len(dados)}.")
    return dados

def plotar(data,dados,y,ajuste,loo,r2,rmse):
    fig,eixos=plt.subplots(1,2,figsize=(10.5,4.6),layout="constrained")
    for eixo,pred,titulo in ((eixos[0],ajuste,"Ajuste"),(eixos[1],loo,"Validação leave-one-out")):
        for cond,estilo in ESTILOS.items():
            m=dados.condicao.eq(cond);eixo.scatter(y[m],pred[m],s=55,edgecolor="white",linewidth=.6,**estilo)
        baixo,alto=min(y.min(),pred.min())-.05,max(y.max(),pred.max())+.05;eixo.plot([baixo,alto],[baixo,alto],"--",color="#475569",linewidth=1)
        eixo.set(xlim=(baixo,alto),ylim=(baixo,alto),xlabel="E observada",ylabel="E predita",title=titulo)
    eixos[0].legend(frameon=False);eixos[1].text(.04,.95,f"n = {len(dados)}\nR² LOO = {r2:.3f}\nRMSE LOO = {rmse:.3f}",transform=eixos[1].transAxes,va="top",bbox={"facecolor":"white","edgecolor":"#94a3b8","boxstyle":"round,pad=.3"})
    fig.suptitle(f"E — {data}: PLSR com Top 5 VIP global");fig.savefig(SAIDA/f"{data.replace('/','_')}_observado_vs_predito.png",dpi=220);plt.close(fig)

def otimizar(x,y):
    espaco={"n_componentes":[1,2,3,4,5],"scale":[True,False]}
    def objetivo(trial):
        pred=prever(x,y,int(trial.suggest_categorical("n_componentes",espaco["n_componentes"])),bool(trial.suggest_categorical("scale",espaco["scale"])));trial.set_user_attr("r2_loo",r2_score(y,pred));return root_mean_squared_error(y,pred)
    estudo=optuna.create_study(direction="minimize",sampler=optuna.samplers.GridSampler(espaco,seed=20260928));estudo.optimize(objetivo,n_trials=10,show_progress_bar=False)
    return int(estudo.best_params["n_componentes"]),bool(estudo.best_params["scale"]),estudo

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING);SAIDA.mkdir(exist_ok=True);dados=carregar();y=dados.e.to_numpy(float);x=dados[BANDAS].to_numpy(float)
    corr=[base.spearman_com_p_exato(x[:,i],y) for i in range(x.shape[1])];tab=pd.DataFrame({"banda_nm":np.asarray(BANDAS,dtype=int),"rho_spearman":[v[0] for v in corr],"p_valor":[v[1] for v in corr]})
    retidas=tab.loc[tab.p_valor<.05,"banda_nm"].astype(str).tolist();selecionadas=ipls.selecionar_ipls_diverso(dados,retidas,y,tab,largura_intervalo_nm=10,limiar_correlacao=None,n_bandas=5).reset_index(drop=True)
    bandas=selecionadas.banda_nm.astype(int).tolist();xf=dados[[str(b) for b in bandas]].to_numpy(float);n_global,scale_global,estudo_global=otimizar(xf,y);modelo_global=PLSRegression(n_components=n_global,scale=scale_global).fit(xf,y);selecionadas["vip_plsr_global"]=base.calcular_vip(modelo_global);top5=selecionadas.sort_values("vip_plsr_global",ascending=False).reset_index(drop=True);top5.insert(0,"rank_vip",range(1,6));top5.to_csv(SAIDA/"top5_vip_global.csv",sep=";",index=False)
    estudo_global.trials_dataframe(attrs=("number","value","params","state","user_attrs")).to_csv(SAIDA/"trials_optuna_global.csv",sep=";",index=False)
    resumo=[]
    for data,parte in dados.groupby("data_fisiologica",sort=False):
        parte=parte.reset_index(drop=True);yd=parte.e.to_numpy(float);xd=parte[[str(b) for b in bandas]].to_numpy(float);n,scale,estudo=otimizar(xd,yd);loo=prever(xd,yd,n,scale);modelo=PLSRegression(n_components=n,scale=scale).fit(xd,yd);ajuste=modelo.predict(xd).ravel();r2,rmse=r2_score(yd,loo),root_mean_squared_error(yd,loo)
        prefixo=data.replace("/","_");parte[["bloco","genotipo","condicao","ordem_repeticao_fisiologia","e"]].assign(predito_ajuste=ajuste,predito_loo=loo).to_csv(SAIDA/f"{prefixo}_predicoes.csv",sep=";",index=False);estudo.trials_dataframe(attrs=("number","value","params","state","user_attrs")).to_csv(SAIDA/f"{prefixo}_trials_optuna.csv",sep=";",index=False);plotar(data,parte,yd,ajuste,loo,r2,rmse)
        resumo.append({"data":data,"n":len(parte),"bandas_top5_vip_global_nm":", ".join(map(str,bandas)),"n_componentes_optuna":n,"scale_optuna":scale,"r2_ajuste":r2_score(yd,ajuste),"rmse_ajuste":root_mean_squared_error(yd,ajuste),"r2_loo":r2,"rmse_loo":rmse})
    pd.DataFrame(resumo).to_csv(SAIDA/"resumo_plsr_diario_top5_vip_global.csv",sep=";",index=False)
    pd.DataFrame([{"n_global":len(dados),"datas":", ".join(DATAS),"p_limiar_spearman":.05,"n_bandas_spearman":len(retidas),"bandas_top5_vip_global_nm":", ".join(map(str,bandas)),"n_componentes_global":n_global,"scale_global":scale_global}]).to_csv(SAIDA/"resumo_selecao_vip_global.csv",sep=";",index=False)
if __name__=="__main__":main()
