#!/usr/bin/env python3
"""iPLS/PLSR para ETR com filtro Spearman, VIP e Top 5 bandas.

Para cada coleta de ETR, as leituras espectrais matinais sao agregadas por
bloco, genotipo e condicao. Bandas com correlacao de Spearman bilateral
significativa com ETR (p < 0,001) sao organizadas em janelas locais de 10 nm;
cada janela e representada pela banda de maior |rho|. O PLSR e ajustado a
essas representantes, retendo VIP >= 1, e avaliado por leave-one-block-out.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
OUT = ROOT / "analiseETR" / "resultados"
P_LIMIAR = 0.001
JANELA_NM = 10
MAX_COMPONENTES = 6
MAPA = {
    "23_02": ("ETR  (23/02)", "D02M", "23/02"),
    "02_03": ("ETR (02/03)", "D09M", "02/03"),
    "03_03": ("ETR (03/03)", "D10M", "03/03"),
}


def calcular_vip(modelo: PLSRegression) -> np.ndarray:
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * ((wn ** 2) * ssy).sum(axis=1) / ssy.sum())


def splits_por_bloco(X: np.ndarray, y: np.ndarray, blocos: np.ndarray):
    return list(GroupKFold(n_splits=len(np.unique(blocos))).split(X, y, blocos))


def escolher_componentes(X: np.ndarray, y: np.ndarray, blocos: np.ndarray):
    folds = splits_por_bloco(X, y, blocos)
    limite = min(MAX_COMPONENTES, X.shape[1], min(len(treino) - 1 for treino, _ in folds))
    linhas = []
    for n_componentes in range(1, limite + 1):
        predito = np.full(len(y), np.nan)
        for treino, teste in folds:
            modelo = PLSRegression(n_components=n_componentes, scale=True).fit(X[treino], y[treino])
            predito[teste] = modelo.predict(X[teste]).ravel()
        linhas.append({
            "n_componentes": n_componentes,
            "r2_cv_bloco": r2_score(y, predito),
            "rmse_cv_bloco": mean_squared_error(y, predito) ** 0.5,
        })
    tabela = pd.DataFrame(linhas)
    return int(tabela.loc[tabela.rmse_cv_bloco.idxmin(), "n_componentes"]), tabela


def predicao_cv(X: np.ndarray, y: np.ndarray, blocos: np.ndarray, n_componentes: int) -> np.ndarray:
    predito = np.full(len(y), np.nan)
    for treino, teste in splits_por_bloco(X, y, blocos):
        modelo = PLSRegression(n_components=n_componentes, scale=True).fit(X[treino], y[treino])
        predito[teste] = modelo.predict(X[teste]).ravel()
    return predito


def preparar_dados():
    fis = pd.read_excel(FISIO)
    # Os dois cabeçalhos do Excel foram salvos com caracteres inválidos; as
    # primeiras colunas são, de forma estável, Genótipo e Condição.
    fis["condicao"] = fis.iloc[:, 1].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["genotipo"] = fis.iloc[:, 0]
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda i: f"B{i + 1}")
    esp = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    bandas = np.array([int(coluna) for coluna in esp.columns if coluna.isdigit()])
    return fis, esp, bandas


def selecionar_iplS(X: np.ndarray, y: np.ndarray, bandas: np.ndarray):
    """Seleciona uma representante por janela de 10 nm entre bandas p < limiar."""
    rhos = np.empty(X.shape[1])
    pvalores = np.empty(X.shape[1])
    for indice in range(X.shape[1]):
        rho, pvalor = spearmanr(X[:, indice], y)
        rhos[indice] = rho
        pvalores[indice] = pvalor
    validas = np.isfinite(rhos) & np.isfinite(pvalores) & (pvalores < P_LIMIAR)
    # iPLS: intervalos fixos de 10 nm, sem misturar regioes espectrais vizinhas.
    inicio = bandas.min()
    intervalos = ((bandas - inicio) // JANELA_NM).astype(int)
    representantes = []
    linhas = []
    for intervalo in np.unique(intervalos[validas]):
        indices = np.flatnonzero((intervalos == intervalo) & validas)
        melhor = indices[np.argmax(np.abs(rhos[indices]))]
        representantes.append(melhor)
        linhas.append({
            "intervalo_inicio_nm": int(inicio + intervalo * JANELA_NM),
            "intervalo_fim_nm": int(inicio + (intervalo + 1) * JANELA_NM - 1),
            "n_bandas_spearman_significativas": len(indices),
            "banda_representante_nm": int(bandas[melhor]),
            "rho_spearman_etr": rhos[melhor],
            "p_spearman_etr": pvalores[melhor],
        })
    detalhe = pd.DataFrame({"banda_nm": bandas, "rho_spearman_etr": rhos, "p_spearman_etr": pvalores,
    "passa_p_limiar": validas, "intervalo_10nm": intervalos})
    return np.array(representantes), pd.DataFrame(linhas), detalhe


def plot_real_predito(y: np.ndarray, ajuste: np.ndarray, cv: np.ndarray, rotulo: str, caminho: Path):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for eixo, predito, titulo in zip(eixos, (ajuste, cv), ("Ajuste", "CV por bloco")):
        r2, rmse = r2_score(y, predito), mean_squared_error(y, predito) ** .5
        minimo, maximo = min(y.min(), predito.min()), max(y.max(), predito.max())
        margem = max((maximo - minimo) * .06, .1)
        eixo.scatter(y, predito, color="#287a8e", edgecolor="white", linewidth=.5, s=48)
        eixo.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem], "--", color="#bd3f32")
        eixo.set(xlabel="ETR real", ylabel="ETR predito", title=f"{titulo}\nR² = {r2:.3f}; RMSE = {rmse:.3f}")
        eixo.set_xlim(minimo - margem, maximo + margem); eixo.set_ylim(minimo - margem, maximo + margem)
    fig.suptitle(f"PLSR ETR — {rotulo}", fontweight="bold")
    fig.tight_layout(); fig.savefig(caminho, dpi=250, bbox_inches="tight"); plt.close(fig)


def metricas_univariadas(X: np.ndarray, y: np.ndarray, blocos: np.ndarray, ranking: pd.DataFrame, rotulo: str, caminho: Path):
    top5 = ranking.head(5).copy().reset_index(drop=True)
    linhas = []
    fig, eixos = plt.subplots(1, len(top5), figsize=(4.0 * len(top5), 3.8), squeeze=False)
    for i, banda in top5.iterrows():
        x = X[:, int(banda["indice_X"])].reshape(-1, 1)
        modelo = LinearRegression().fit(x, y)
        ajuste = modelo.predict(x)
        cv = np.full(len(y), np.nan)
        for treino, teste in splits_por_bloco(x, y, blocos):
            cv[teste] = LinearRegression().fit(x[treino], y[treino]).predict(x[teste])
        r2_ajuste, rmse_ajuste = r2_score(y, ajuste), mean_squared_error(y, ajuste) ** .5
        r2_cv, rmse_cv = r2_score(y, cv), mean_squared_error(y, cv) ** .5
        linhas.append({"data_etr": rotulo, "ordem_vip": i + 1, "banda_nm": int(banda.banda_nm),
                       "vip_modelo_final": banda.vip_modelo_final, "rho_spearman_etr": banda.rho_spearman_etr,
                       "r2_ajuste": r2_ajuste, "rmse_ajuste": rmse_ajuste,
                       "r2_cv_bloco": r2_cv, "rmse_cv_bloco": rmse_cv})
        eixo = eixos[0, i]
        eixo.scatter(x.ravel(), y, color="#287a8e", edgecolor="white", linewidth=.4, s=36)
        ordem = np.argsort(x.ravel())
        eixo.plot(x.ravel()[ordem], ajuste[ordem], color="#bd3f32", lw=1.5)
        eixo.set(title=f"{int(banda.banda_nm)} nm\nR²={r2_ajuste:.3f}; RMSE={rmse_ajuste:.3f}", xlabel="Reflectância", ylabel="ETR")
    fig.suptitle(f"Top 5 VIP: reflectância × ETR — {rotulo}", fontweight="bold")
    fig.tight_layout(); fig.savefig(caminho, dpi=250, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(linhas)


def analisar(nome, coluna_etr, dia, rotulo, fis, esp, bandas):
    meta = ["bloco", "genotipo", "condicao"]
    medias = esp[(esp.data_coleta == dia) & (esp.turno == "manha")].groupby(meta, as_index=False)[[str(b) for b in bandas]].mean()
    dados = fis[meta + [coluna_etr]].merge(medias, on=meta, how="inner", validate="one_to_one").dropna(subset=[coluna_etr]).reset_index(drop=True)
    if len(dados) < 8:
        raise ValueError(f"{rotulo}: amostras validas insuficientes ({len(dados)}).")
    y = dados[coluna_etr].to_numpy(float)
    X = dados[[str(b) for b in bandas]].to_numpy(float)
    blocos = dados.bloco.to_numpy()
    reps, intervalos, spearman = selecionar_iplS(X, y, bandas)
    if not len(reps):
        spearman.to_csv(OUT / f"{nome}_spearman_todas_bandas.csv", sep=";", index=False)
        return {"data_etr": rotulo, "coleta_espectral": dia, "n": len(y), "bandas_iniciais": len(bandas),
                "bandas_spearman_p_lt_limiar": 0, "intervalos_iplS_10nm": 0,
                "bandas_vip_ge_1": 0, "componentes": np.nan,
                "r2_ajuste": np.nan, "rmse_ajuste": np.nan,
                "r2_cv_bloco": np.nan, "rmse_cv_bloco": np.nan,
                "status": f"Sem bandas com Spearman p < {P_LIMIAR:g}; PLSR/VIP não ajustado."}
    Xr, br = X[:, reps], bandas[reps]
    n_inicial, _ = escolher_componentes(Xr, y, blocos)
    vip_inicial = calcular_vip(PLSRegression(n_components=n_inicial, scale=True).fit(Xr, y))
    manter = vip_inicial >= 1
    if not manter.any():
        manter[np.argmax(vip_inicial)] = True
    Xfinal, bandas_finais = Xr[:, manter], br[manter]
    n_final, componentes = escolher_componentes(Xfinal, y, blocos)
    modelo = PLSRegression(n_components=n_final, scale=True).fit(Xfinal, y)
    vip_final = calcular_vip(modelo)
    ajuste = modelo.predict(Xfinal).ravel()
    cv = predicao_cv(Xfinal, y, blocos, n_final)
    indice_original = reps[manter]
    ranking = pd.DataFrame({"banda_nm": bandas_finais, "vip_selecao": vip_inicial[manter],
                            "vip_modelo_final": vip_final, "rho_spearman_etr": spearman.loc[indice_original, "rho_spearman_etr"].to_numpy(),
                            "p_spearman_etr": spearman.loc[indice_original, "p_spearman_etr"].to_numpy(), "indice_X": indice_original})
    ranking = ranking.sort_values("vip_modelo_final", ascending=False).reset_index(drop=True)
    ranking.insert(0, "data_etr", rotulo); ranking.insert(1, "coleta_espectral", dia)
    ranking.to_csv(OUT / f"{nome}_bandas_vip_ge_1.csv", sep=";", index=False)
    intervalos.to_csv(OUT / f"{nome}_intervalos_iplS_10nm.csv", sep=";", index=False)
    spearman.to_csv(OUT / f"{nome}_spearman_todas_bandas.csv", sep=";", index=False)
    componentes.to_csv(OUT / f"{nome}_componentes_cv.csv", sep=";", index=False)
    dados[meta + [coluna_etr]].assign(predito_ajuste=ajuste, predito_cv_bloco=cv, residuo_cv_bloco=y - cv).to_csv(OUT / f"{nome}_predicoes.csv", sep=";", index=False)
    plot_real_predito(y, ajuste, cv, rotulo, OUT / f"{nome}_real_x_predito.png")
    top5 = metricas_univariadas(X, y, blocos, ranking, rotulo, OUT / f"{nome}_top5_vip_reflectancia_x_etr.png")
    top5.to_csv(OUT / f"{nome}_top5_vip_reflectancia_x_etr.csv", sep=";", index=False)
    return {"data_etr": rotulo, "coleta_espectral": dia, "n": len(y), "bandas_iniciais": len(bandas),
            "bandas_spearman_p_lt_limiar": int(spearman.passa_p_limiar.sum()), "intervalos_iplS_10nm": len(intervalos),
            "bandas_vip_ge_1": int(manter.sum()), "componentes": n_final,
            "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": mean_squared_error(y, ajuste) ** .5,
            "r2_cv_bloco": r2_score(y, cv), "rmse_cv_bloco": mean_squared_error(y, cv) ** .5,
            "status": "Modelo ajustado"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fis, esp, bandas = preparar_dados()
    resultados = [analisar(nome, *args, fis, esp, bandas) for nome, args in MAPA.items()]
    resumo = pd.DataFrame(resultados)
    resumo.to_csv(OUT / "resumo_plsr_etr.csv", sep=";", index=False)
    p_texto = f"{P_LIMIAR:g}".replace(".", ",")
    linhas = ["# iPLS/PLSR — ETR", "", "Pareamento: média das leituras espectrais matinais por bloco × genótipo × condição; D02M=23/02, D09M=02/03 e D10M=03/03.", "", f"Em cada data, calcula-se Spearman bilateral entre cada banda e ETR; somente p < {p_texto} entra. As bandas aprovadas são separadas em intervalos fixos de 10 nm (iPLS), dos quais se mantém a maior |rho|. O PLSR escalonado escolhe os componentes pelo menor RMSE em validação cruzada deixando um bloco de fora; VIP ≥ 1,0 é retido no modelo final.", "", f"Quando nenhuma banda atende p < {p_texto}, o PLSR/VIP não é ajustado — não há substituição do limiar. As métricas CV são diagnósticas: Spearman/iPLS/VIP foram definidos antes das dobras e, portanto, ainda podem ser otimistas. Os gráficos Top 5 mostram regressões lineares univariadas reflectância × ETR; o CSV contém métricas de ajuste e CV por bloco.", "", f"| ETR | Coleta espectral | n | Bandas Spearman p < {p_texto} | Intervalos iPLS | VIP ≥1 | Componentes | R² ajuste | RMSE ajuste | R² CV-bloco | RMSE CV-bloco | Status |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in resultados:
        def fmt(valor): return "—" if pd.isna(valor) else f"{valor:.3f}"
        linhas.append(f"| {r['data_etr']} | {r['coleta_espectral']} | {r['n']} | {r['bandas_spearman_p_lt_limiar']} | {r['intervalos_iplS_10nm']} | {r['bandas_vip_ge_1']} | {fmt(r['componentes'])} | {fmt(r['r2_ajuste'])} | {fmt(r['rmse_ajuste'])} | {fmt(r['r2_cv_bloco'])} | {fmt(r['rmse_cv_bloco'])} | {r['status']} |")
    (OUT / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
