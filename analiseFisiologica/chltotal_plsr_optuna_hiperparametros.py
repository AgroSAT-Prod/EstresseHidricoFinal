#!/usr/bin/env python3
"""Busca Optuna dos hiperparâmetros do PLSR para CHL Total.

As cinco bandas obtidas por Spearman (p < 0,001) + iPLS (10 nm) permanecem
fixas. São avaliados todos os valores úteis de ``n_components`` e as duas
opções de ``scale`` do PLSRegression; ``tol`` e ``max_iter`` só controlam a
convergência numérica do algoritmo e não são parâmetros preditivos neste
modelo de resposta univariada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import clorofila_total_spearman_plsr as base
import cra_interval_pls_vip as ipls


SAIDA = Path(__file__).resolve().parent / "resultados_chltotal_plsr_optuna_hiperparametros"
SEED = 20260928


def loo(x: np.ndarray, y: np.ndarray, componentes: int, scale: bool) -> np.ndarray:
    return cross_val_predict(
        PLSRegression(n_components=componentes, scale=scale, tol=1e-6, max_iter=500),
        x, y, cv=LeaveOneOut(),
    ).ravel()


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    fisiologia = base.ler_fisiologia()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisiologia, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)
    x = dados[bandas].to_numpy(float)
    resultados_spearman = [base.spearman_com_p_exato(x[:, indice], y) for indice in range(x.shape[1])]
    tabela = pd.DataFrame({
        "banda_nm": np.asarray(bandas, dtype=int),
        "rho_spearman": [resultado[0] for resultado in resultados_spearman],
        "p_valor": [resultado[1] for resultado in resultados_spearman],
    })
    escolhidas = ipls.selecionar_ipls_diverso(
        dados, tabela.loc[tabela["p_valor"] < .001, "banda_nm"].astype(str).tolist(), y, tabela,
        largura_intervalo_nm=10, limiar_correlacao=None, n_bandas=5,
    )
    bandas_finais = escolhidas["banda_nm"].astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)

    espaco = {"n_componentes": [1, 2, 3, 4, 5], "scale": [True, False]}
    def objetivo(trial: optuna.Trial) -> float:
        componentes = trial.suggest_categorical("n_componentes", espaco["n_componentes"])
        scale = trial.suggest_categorical("scale", espaco["scale"])
        predicao = loo(x_final, y, int(componentes), bool(scale))
        trial.set_user_attr("r2_loo", r2_score(y, predicao))
        return root_mean_squared_error(y, predicao)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=SEED))
    estudo.optimize(objetivo, n_trials=10, show_progress_bar=False)
    parametros = estudo.best_params
    n_componentes, scale = int(parametros["n_componentes"]), bool(parametros["scale"])
    predicao = loo(x_final, y, n_componentes, scale)
    modelo = PLSRegression(n_components=n_componentes, scale=scale, tol=1e-6, max_iter=500).fit(x_final, y)
    ajuste = modelo.predict(x_final).ravel()

    # O PLSR originalmente reportado usa scale=True e cinco componentes.
    predicao_base = loo(x_final, y, 5, True)
    comparacao = pd.DataFrame([
        {"modelo": "PLSR base", "n_componentes": 5, "scale": True,
         "r2_loo": r2_score(y, predicao_base), "rmse_loo": root_mean_squared_error(y, predicao_base)},
        {"modelo": "PLSR otimizado por Optuna", "n_componentes": n_componentes, "scale": scale,
         "r2_loo": r2_score(y, predicao), "rmse_loo": root_mean_squared_error(y, predicao)},
    ])
    comparacao["delta_r2_vs_base"] = comparacao["r2_loo"] - comparacao.loc[0, "r2_loo"]
    comparacao["delta_rmse_vs_base"] = comparacao["rmse_loo"] - comparacao.loc[0, "rmse_loo"]
    comparacao.to_csv(SAIDA / "comparacao_metricas_loo.csv", sep=";", index=False)
    estudo.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(
        SAIDA / "trials_optuna_hiperparametros.csv", sep=";", index=False
    )
    pd.DataFrame([{
        "n_trials": len(estudo.trials), "seed": SEED, "bandas_nm": ", ".join(map(str, bandas_finais)),
        "n_componentes": n_componentes, "scale": scale, "tol": 1e-6, "max_iter": 500,
        "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": root_mean_squared_error(y, ajuste),
        "r2_loo": r2_score(y, predicao), "rmse_loo": root_mean_squared_error(y, predicao),
    }]).to_csv(SAIDA / "melhores_hiperparametros.csv", sep=";", index=False)
    print(comparacao.to_string(index=False))


if __name__ == "__main__":
    main()
