#!/usr/bin/env python3
"""Busca ampla de PLSR para os dois painéis fisiológicos prioritários.

O objetivo é o R² das predições fora da dobra em GroupKFold, deixando um dos
quatro blocos de campo de fora. A mesma seleção (Spearman -> VIP) é aplicada
antes das dobras, como nos gráficos originais; portanto, esta é uma busca
comparável aos resultados anteriores, não uma validação externa aninhada.
"""
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ESPECTROS = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
CONFIGS = {
    "chltotal_25_02": {
        "out": ROOT / "analiseClorofilaTotal" / "resultados",
        "arquivo": "25_02",
        "alvo": "CHL TOTAL (25.02)",
        "dia": "D04M",
        "rotulo": "25/02",
        "col_data": "data_clorofila_total",
        "col_rho": "rho_spearman_clorofila_total",
        "r2_anterior": 0.5306162312005123,
    },
    "cra_02_03": {
        "out": ROOT / "analiseCRA" / "resultados",
        "arquivo": "02_03",
        "alvo": "CRA (02/03)",
        "dia": "D09M",
        "rotulo": "02/03",
        "col_data": "data_cra",
        "col_rho": "rho_spearman_cra",
        "r2_anterior": 0.7710516170237192,
    },
}
N_TRIALS = 400
SEED = 20260924


def vip(modelo):
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    ssy = (q**2).ravel() * (t**2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * ((wn**2) * ssy).sum(axis=1) / ssy.sum())


def carregar(config):
    fis = pd.read_excel(FISIO)
    fis["condicao"] = fis["Condição"].replace({"IRR": "IRRIG", "NIR": "NIRRIG"})
    fis["genotipo"] = fis["Genótipo"]
    fis["bloco"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount().map(lambda n: f"B{n + 1}")
    esp = pd.read_csv(ESPECTROS, sep=";", decimal=",")
    bandas = np.array([int(col) for col in esp.columns if col.isdigit()])
    meta = ["bloco", "genotipo", "condicao"]
    medias = esp.loc[(esp.data_coleta == config["dia"]) & (esp.turno == "manha")].groupby(
        meta, as_index=False
    )[[str(b) for b in bandas]].mean()
    dados = fis[meta + [config["alvo"]]].merge(medias, on=meta, how="inner", validate="one_to_one")
    dados = dados.dropna(subset=[config["alvo"]]).reset_index(drop=True)
    X = dados[[str(b) for b in bandas]].to_numpy(float)
    y = dados[config["alvo"]].to_numpy(float)
    return dados, X, y, dados["bloco"].to_numpy(), bandas


def selecionar(X, y, bandas, corr, limiar_r, janela_nm, n_pre, limiar_vip):
    grupos_rho = agrupar(corr, bandas, limiar_r=limiar_r, janela_nm=janela_nm)
    ry = np.array([pd.Series(X[:, j]).corr(pd.Series(y), method="spearman") for j in range(X.shape[1])])
    reps = []
    for grupo in np.unique(grupos_rho):
        idx = np.flatnonzero(grupos_rho == grupo)
        reps.append(idx[np.nanargmax(np.abs(ry[idx]))])
    reps = np.array(reps)
    Xr = X[:, reps]
    n_pre = min(n_pre, Xr.shape[1], Xr.shape[0] - 1)
    modelo_pre = PLSRegression(n_components=n_pre, scale=True).fit(Xr, y)
    mascara = vip(modelo_pre) >= limiar_vip
    if not mascara.any():
        mascara[np.nanargmax(vip(modelo_pre))] = True
    return Xr[:, mascara], bandas[reps][mascara], ry[reps][mascara], vip(modelo_pre)[mascara], len(reps)


def predicao_cv(X, y, grupos, n_componentes):
    pred = np.full(len(y), np.nan)
    for treino, teste in GroupKFold(n_splits=len(np.unique(grupos))).split(X, y, grupos):
        modelo = PLSRegression(n_components=n_componentes, scale=True).fit(X[treino], y[treino])
        pred[teste] = modelo.predict(X[teste]).ravel()
    return pred


def executar(config):
    dados, X, y, grupos, bandas = carregar(config)
    corr = spearman_matriz(X)
    cache = {}

    def objetivo(trial):
        limiar_r = trial.suggest_float("limiar_spearman", 0.55, 0.95)
        janela_nm = trial.suggest_int("janela_nm", 3, 35)
        n_pre = trial.suggest_int("componentes_pre", 1, 8)
        limiar_vip = trial.suggest_float("limiar_vip", 0.50, 1.50)
        try:
            Xs, _, _, _, _ = selecionar(X, y, bandas, corr, limiar_r, janela_nm, n_pre, limiar_vip)
            limite = min(8, Xs.shape[1], min(len(idx) - 1 for idx, _ in GroupKFold(n_splits=4).split(Xs, y, grupos)))
            if limite < 1:
                raise ValueError("nenhuma banda elegível")
            n_final = trial.suggest_int("componentes_finais", 1, limite)
            pred = predicao_cv(Xs, y, grupos, n_final)
            cache[trial.number] = (Xs.shape[1], r2_score(y, pred), mean_squared_error(y, pred) ** .5)
            return r2_score(y, pred)
        except Exception as exc:
            trial.set_user_attr("erro", str(exc))
            raise optuna.TrialPruned()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50),
    )
    study.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=False)
    melhores = study.best_params
    Xs, bs, rhos, vips_pre, n_reps = selecionar(
        X, y, bandas, corr, melhores["limiar_spearman"], melhores["janela_nm"],
        melhores["componentes_pre"], melhores["limiar_vip"],
    )
    n_final = melhores["componentes_finais"]
    pred = predicao_cv(Xs, y, grupos, n_final)
    modelo = PLSRegression(n_components=n_final, scale=True).fit(Xs, y)
    vips_final = vip(modelo)
    r2_novo = r2_score(y, pred)
    rmse_novo = mean_squared_error(y, pred) ** .5
    out = config["out"]
    out.mkdir(parents=True, exist_ok=True)
    prefixo = config["arquivo"]
    ranking = pd.DataFrame({
        "banda_nm": bs, "vip_selecao": vips_pre, "vip_modelo_final": vips_final, config["col_rho"]: rhos,
    }).sort_values("vip_modelo_final", ascending=False)
    ranking.insert(0, config["col_data"], config["rotulo"])
    ranking.insert(1, "coleta_espectral", config["dia"])
    ranking.to_csv(out / f"{prefixo}_bandas_vip_ge_1.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", config["alvo"]]].assign(
        predito_cv=pred, residuo_cv=y - pred
    ).to_csv(out / f"{prefixo}_predicoes_cv.csv", sep=";", index=False)
    pd.DataFrame([{
        "n_componentes": n_final, "r2_cv_bloco": r2_novo, "rmse_cv_bloco": rmse_novo,
        "n_bandas": Xs.shape[1], "representantes_spearman": n_reps,
    }]).to_csv(out / f"{prefixo}_componentes_cv.csv", sep=";", index=False)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.to_csv(out / f"{prefixo}_optuna_amplo_trials.csv", sep=";", index=False)
    parametros = pd.DataFrame([{
        "alvo": config["alvo"], "n_trials": N_TRIALS, "seed": SEED, "r2_cv_anterior": config["r2_anterior"],
        "r2_cv_optuna": r2_novo, "delta_r2": r2_novo - config["r2_anterior"], "rmse_cv_optuna": rmse_novo,
        "n_amostras": len(y), "n_bandas_finais": Xs.shape[1], "representantes_spearman": n_reps, **melhores,
    }])
    parametros.to_csv(out / f"{prefixo}_optuna_amplo_melhor_configuracao.csv", sep=";", index=False)
    print(f"{config['alvo']}: R² CV {config['r2_anterior']:.3f} -> {r2_novo:.3f}; {Xs.shape[1]} bandas; {n_final} componentes")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for config in CONFIGS.values():
        executar(config)


if __name__ == "__main__":
    main()
