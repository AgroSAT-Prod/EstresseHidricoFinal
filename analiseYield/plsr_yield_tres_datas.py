#!/usr/bin/env python3
"""PLSR para Yield nas tres coletas fisiologicas.

As planilhas fisiologicas contem quatro repeticoes, sem a coluna ``bloco``.
A ordem de cada combinacao genotipo × condicao e usada como B1--B4, que e a
ordem do delineamento e permite parear cada repeticao com a media das oito
leituras espectrais do bloco. As coletas sao D02M=23/02, D09M=02/03 e
D10M=03/03.

Em cada data, bandas localmente redundantes sao agrupadas por Spearman
|rho| >= 0,80 (ligacao completa, janela menor que 10 nm). A representante de
cada grupo e a banda com maior |rho| com Yield. O PLSR e ajustado com essas
representantes; mantem-se VIP >= 1,0 e se reajusta o modelo final.
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
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
OUT = ROOT / "analiseYield" / "resultados"
MAPA = {
    "23_02": ("Yield (23/02)", "D02M", "23/02"),
    "02_03": ("Yield (02/03)", "D09M", "02/03"),
    "03_03": ("Yield (03.03)", "D10M", "03/03"),
}
MAX_COMPONENTES = 6


def vip(modelo):
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * ((wn ** 2) * ssy).sum(axis=1) / ssy.sum())


def cv_componentes(X, y, grupos):
    folds = list(GroupKFold(n_splits=len(np.unique(grupos))).split(X, y, grupos))
    limite = min(MAX_COMPONENTES, X.shape[1], min(len(tr) - 1 for tr, _ in folds))
    linhas = []
    for n in range(1, limite + 1):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
        linhas.append({"n_componentes": n, "rmse_cv_bloco": mean_squared_error(y, pred) ** .5,
                       "r2_cv_bloco": r2_score(y, pred)})
    tab = pd.DataFrame(linhas)
    return int(tab.loc[tab.rmse_cv_bloco.idxmin(), "n_componentes"]), tab


def predicao_cv(X, y, grupos, n):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=len(np.unique(grupos))).split(X, y, grupos):
        pred[te] = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr]).predict(X[te]).ravel()
    return pred


def preparar():
    fis = pd.read_excel(FISIO)
    fis["condicao"] = fis["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["genotipo"] = fis["Genótipo"]
    # Cada grupo no Excel esta ordenado nas quatro repeticoes do delineamento.
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda i: f"B{i + 1}")
    esp = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    bandas = np.array([int(c) for c in esp.columns if c.isdigit()])
    return fis, esp, bandas


def analisar(nome, coluna_y, dia, rotulo, fis, esp, bandas):
    meta = ["bloco", "genotipo", "condicao"]
    sub = esp[(esp.data_coleta == dia) & (esp.turno == "manha")]
    medias = sub.groupby(meta, as_index=False)[[str(b) for b in bandas]].mean()
    dados = fis[meta + [coluna_y]].merge(medias, on=meta, how="inner", validate="one_to_one")
    if len(dados) != 24:
        raise ValueError(f"{rotulo}: pareamento incompleto ({len(dados)}/24)")
    # Algumas variáveis fisiológicas têm ausência pontual; a unidade sem alvo
    # não pode entrar no ajuste nem na validação cruzada daquele dia.
    dados = dados.dropna(subset=[coluna_y]).reset_index(drop=True)
    if len(dados) < 8:
        raise ValueError(f"{rotulo}: amostras válidas insuficientes ({len(dados)})")
    y = dados[coluna_y].to_numpy(float)
    X = dados[[str(b) for b in bandas]].to_numpy(float)
    grupos = dados.bloco.to_numpy()

    corr_x = spearman_matriz(X)
    grupos_rho = agrupar(corr_x, bandas, limiar_r=.80, janela_nm=10.0)
    # Spearman de cada banda com Yield; o maior valor absoluto representa seu grupo.
    ry = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = []
    for g in np.unique(grupos_rho):
        idx = np.flatnonzero(grupos_rho == g)
        reps.append(idx[np.nanargmax(np.abs(ry[idx]))])
    reps = np.array(reps)
    Xr, br = X[:, reps], bandas[reps]
    n_pre, tab_pre = cv_componentes(Xr, y, grupos)
    pre = PLSRegression(n_components=n_pre, scale=True).fit(Xr, y)
    vip_pre = vip(pre)
    selecionadas = vip_pre >= 1.0
    # Degeneracao rara: garante ao menos a banda mais importante.
    if not selecionadas.any():
        selecionadas[np.nanargmax(vip_pre)] = True
    Xs, bs = Xr[:, selecionadas], br[selecionadas]
    n_final, tab_final = cv_componentes(Xs, y, grupos)
    modelo = PLSRegression(n_components=n_final, scale=True).fit(Xs, y)
    vfinal = vip(modelo)
    ajuste = modelo.predict(Xs).ravel()
    cv = predicao_cv(Xs, y, grupos, n_final)
    ranking = pd.DataFrame({"banda_nm": bs, "vip_selecao": vip_pre[selecionadas],
                            "vip_modelo_final": vfinal,
                            "rho_spearman_yield": ry[reps][selecionadas]}).sort_values("vip_modelo_final", ascending=False)
    ranking.insert(0, "data_yield", rotulo)
    ranking.insert(1, "coleta_espectral", dia)
    ranking.to_csv(OUT / f"{nome}_bandas_vip_ge_1.csv", sep=";", index=False)
    tab_final.to_csv(OUT / f"{nome}_componentes_cv.csv", sep=";", index=False)
    dados[meta + [coluna_y]].assign(predito_cv=cv, residuo_cv=y-cv).to_csv(OUT / f"{nome}_predicoes_cv.csv", sep=";", index=False)

    plt.figure(figsize=(8, max(3, min(8, len(ranking) * .25))))
    top = ranking.head(20).sort_values("vip_modelo_final")
    plt.barh(top.banda_nm.astype(str), top.vip_modelo_final, color="#287a8e")
    plt.axvline(1, color="#bd3f32", ls="--", lw=1, label="VIP = 1,0")
    plt.xlabel("VIP"); plt.ylabel("Banda (nm)")
    plt.title(f"Yield {rotulo}: bandas VIP ≥ 1,0")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / f"{nome}_top_vip.png", dpi=220); plt.close()
    return {"data_yield": rotulo, "coleta_espectral": dia, "n": len(y), "bandas_iniciais": len(bandas),
            "representantes_spearman": len(br), "bandas_vip_ge_1": int(selecionadas.sum()),
            "componentes": n_final, "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": mean_squared_error(y, ajuste) ** .5,
            "r2_cv_bloco": r2_score(y, cv), "rmse_cv_bloco": mean_squared_error(y, cv) ** .5}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fis, esp, bandas = preparar()
    resultados = [analisar(nome, *args, fis, esp, bandas) for nome, args in MAPA.items()]
    resumo = pd.DataFrame(resultados)
    resumo.to_csv(OUT / "resumo_plsr_yield.csv", sep=";", index=False)
    linhas = ["# PLSR — Yield nas três coletas", "", "Pareamento: média de leituras espectrais matinais por bloco × genótipo × condição; D02M=23/02, D09M=02/03 e D10M=03/03.", "", "Spearman |ρ| ≥ 0,80 foi aplicado entre bandas contíguas (janela <10 nm). Em cada grupo, manteve-se a banda de maior |ρ| com Yield. O PLSR foi escalonado, com componentes escolhidos pelo menor RMSE em validação cruzada deixando um bloco de fora; VIP ≥1,0 do ajuste inicial foi retido e o modelo final reajustado.", "", "`R² CV-bloco` é diagnóstico: a seleção de representantes/VIP foi feita antes das dobras; portanto, a estimativa pode ser otimista e não substitui uma seleção aninhada em nova amostra.", "", "| Yield | Coleta espectral | n | Rep. Spearman | VIP ≥1 | Componentes | R² ajuste | R² CV-bloco | RMSE CV |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in resultados:
        linhas.append(f"| {r['data_yield']} | {r['coleta_espectral']} | {r['n']} | {r['representantes_spearman']} | {r['bandas_vip_ge_1']} | {r['componentes']} | {r['r2_ajuste']:.3f} | {r['r2_cv_bloco']:.3f} | {r['rmse_cv_bloco']:.4f} |")
    (OUT / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
