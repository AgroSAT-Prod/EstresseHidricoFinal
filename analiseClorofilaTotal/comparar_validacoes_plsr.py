#!/usr/bin/env python3
"""Compara estratégias de validação para os PLSR já selecionados de CHL Total.

As bandas VIP e o número de componentes vêm dos ajustes descritivos anteriores.
Assim, os R² desta comparação avaliam somente a estratégia de particionamento;
não são uma validação aninhada da seleção Spearman/VIP.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, ShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analiseYield"))
import plsr_yield_tres_datas as base  # noqa: E402

OUT = ROOT / "analiseClorofilaTotal" / "resultados"
DATAS = {
    "23_02": ("CHL TOTAL (23/02)", "D02M", "23/02"),
    "24_02": ("CHL TOTAL (24/02)", "D03M", "24/02"),
    "25_02": ("CHL TOTAL (25.02)", "D04M", "25/02"),
    "26_02": ("CHL TOTAL (26/02)", "D05M", "26/02"),
    "27_02": ("CHL TOTAL (27/02)", "D06M", "27/02"),
    "02_03": ("CHL TOTAL (02/03)", "D09M", "02/03"),
    "03_03": ("CHL TOTAL (03/03)", "D10M", "03/03"),
}
N_REPETICOES = 100
SEMENTE = 42


def ajustar_prever(X, y, treino, teste, n_comp):
    n = min(n_comp, X.shape[1], len(treino) - 1)
    return PLSRegression(n_components=n, scale=True).fit(X[treino], y[treino]).predict(X[teste]).ravel()


def r2_repetido(X, y, grupos, n_comp, estrategia):
    if estrategia == "CV=4":
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=4).split(X, y, grupos):
            pred[te] = ajustar_prever(X, y, tr, te, n_comp)
        return np.array([r2_score(y, pred)])
    valores = []
    for rep in range(N_REPETICOES):
        if estrategia == "70/30":
            divisao = ShuffleSplit(n_splits=1, test_size=.30, random_state=SEMENTE + rep)
        elif estrategia == "80/20":
            divisao = ShuffleSplit(n_splits=1, test_size=.20, random_state=SEMENTE + rep)
        elif estrategia == "CV=5":
            divisao = KFold(n_splits=5, shuffle=True, random_state=SEMENTE + rep)
        else:
            raise ValueError(estrategia)
        if estrategia.startswith("CV"):
            pred = np.full(len(y), np.nan)
            for tr, te in divisao.split(X):
                pred[te] = ajustar_prever(X, y, tr, te, n_comp)
            valores.append(r2_score(y, pred))
        else:
            tr, te = next(divisao.split(X))
            valores.append(r2_score(y[te], ajustar_prever(X, y, tr, te, n_comp)))
    return np.array(valores)


def r2_loo(X, y, n_comp):
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneOut().split(X):
        pred[te] = ajustar_prever(X, y, tr, te, n_comp)
    return r2_score(y, pred)


def dados_data(fis, esp, bandas, alvo, dia, rotulo, codigo):
    meta = ["bloco", "genotipo", "condicao"]
    sub = esp[(esp.data_coleta == dia) & (esp.turno == "manha")]
    medias = sub.groupby(meta, as_index=False)[[str(b) for b in bandas]].mean()
    dados = fis[meta + [alvo]].merge(medias, on=meta, how="inner", validate="one_to_one").dropna(subset=[alvo])
    ranking = pd.read_csv(OUT / f"{codigo}_bandas_vip_ge_1.csv", sep=";")
    bandas_sel = ranking.banda_nm.astype(int).astype(str).tolist()
    n_comp = int(pd.read_csv(OUT / "resumo_plsr_clorofila_total.csv", sep=";").query("data_yield == @rotulo").componentes.iloc[0])
    return dados[bandas_sel].to_numpy(float), dados[alvo].to_numpy(float), dados["bloco"].to_numpy(), n_comp


def main():
    fis, esp, bandas = base.preparar()
    resumo, distribuicoes = [], []
    for codigo, (alvo, dia, rotulo) in DATAS.items():
        X, y, grupos, n_comp = dados_data(fis, esp, bandas, alvo, dia, rotulo, codigo)
        for estrategia in ("70/30", "80/20", "CV=4", "CV=5"):
            r2 = r2_repetido(X, y, grupos, n_comp, estrategia)
            distribuicoes.extend({"data": rotulo, "estrategia": estrategia, "repeticao": i + 1, "r2": v} for i, v in enumerate(r2))
            resumo.append({"data": rotulo, "estrategia": estrategia, "n": len(y),
                           "r2_medio": r2.mean(), "r2_dp": r2.std(ddof=1),
                           "r2_mediana": np.median(r2), "r2_p025": np.quantile(r2, .025), "r2_p975": np.quantile(r2, .975)})
        r2 = r2_loo(X, y, n_comp)
        resumo.append({"data": rotulo, "estrategia": "LOO", "n": len(y), "r2_medio": r2,
                       "r2_dp": np.nan, "r2_mediana": r2, "r2_p025": np.nan, "r2_p975": np.nan})
    tabela = pd.DataFrame(resumo)
    tabela.to_csv(OUT / "comparacao_r2_validacoes.csv", sep=";", index=False)
    pd.DataFrame(distribuicoes).to_csv(OUT / "comparacao_r2_validacoes_repeticoes.csv", sep=";", index=False)
    linhas = ["# Comparação de validações — PLSR de Clorofila Total", "", "70/30, 80/20 e CV=5 usam 100 particionamentos aleatórios (sementes 42–141); a tabela traz R² médio ± DP. LOO e CV=4 são únicos; CV=4 mantém os quatro blocos do experimento como grupos. CV=5 é KFold aleatório, pois não há cinco blocos. Bandas VIP e componentes foram mantidos fixos, selecionados antes das dobras; resultados são diagnósticos.", "", "| Data | 70/30 R² | 80/20 R² | LOO R² | CV=5 R² | CV=4 (blocos) R² |", "|---|---:|---:|---:|---:|---:|"]
    for data in [v[2] for v in DATAS.values()]:
        d = tabela[tabela.data.eq(data)].set_index("estrategia")
        formata = lambda e: f"{d.loc[e, 'r2_medio']:.3f} ± {d.loc[e, 'r2_dp']:.3f}" if e not in ("LOO", "CV=4") else f"{d.loc[e, 'r2_medio']:.3f}"
        linhas.append(f"| {data} | {formata('70/30')} | {formata('80/20')} | {formata('LOO')} | {formata('CV=5')} | {formata('CV=4')} |")
    (OUT / "comparacao_r2_validacoes.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
