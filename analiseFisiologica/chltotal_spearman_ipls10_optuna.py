#!/usr/bin/env python3
"""Otimiza o PLSR de CHL Total após Spearman (p < 0,001) e iPLS de 10 nm.

Mantém as etapas de seleção espectral do pipeline-base. O estudo Optuna procura
o número de bandas iPLS (1--5) e de componentes PLS (1--5), minimizando o RMSE
das predições leave-one-out. Isso permite uma comparação direta com o PLSR
original de cinco bandas.
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


SAIDA = Path(__file__).resolve().parent / "resultados_chltotal_spearman_p0001_ipls10_optuna"
P_LIMIAR = 0.001
LARGURA_JANELA_NM = 10
SEED = 20260928


def predizer_loo(x: np.ndarray, y: np.ndarray, n_componentes: int) -> np.ndarray:
    return cross_val_predict(
        PLSRegression(n_components=n_componentes, scale=True), x, y, cv=LeaveOneOut()
    ).ravel()


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    fisiologia = base.ler_fisiologia()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(
        fisiologia, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one"
    )
    y = dados["chl_total"].to_numpy(float)
    x_todas = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x_todas[:, i], y) for i in range(x_todas.shape[1])]
    tabela = pd.DataFrame({
        "banda_nm": np.array(bandas, dtype=int),
        "rho_spearman": [resultado[0] for resultado in correlacoes],
        "p_valor": [resultado[1] for resultado in correlacoes],
    })
    retidas = tabela.loc[tabela["p_valor"] < P_LIMIAR, "banda_nm"].astype(str).tolist()
    escolhidas = ipls.selecionar_ipls_diverso(
        dados, retidas, y, tabela, largura_intervalo_nm=LARGURA_JANELA_NM,
        limiar_correlacao=None, n_bandas=5,
    ).reset_index(drop=True)

    # Cada combinação válida é avaliada uma vez: busca exaustiva discreta via Optuna.
    espaco = {
        "n_bandas_ipls": [1, 2, 3, 4, 5],
        "n_componentes": [1, 2, 3, 4, 5],
    }
    def objetivo(trial: optuna.Trial) -> float:
        n_bandas = trial.suggest_categorical("n_bandas_ipls", espaco["n_bandas_ipls"])
        n_componentes = trial.suggest_categorical("n_componentes", espaco["n_componentes"])
        if n_componentes > n_bandas:
            raise optuna.TrialPruned("componentes não podem exceder bandas")
        colunas = escolhidas.head(n_bandas)["banda_nm"].astype(str).tolist()
        predicao = predizer_loo(dados[colunas].to_numpy(float), y, n_componentes)
        rmse = root_mean_squared_error(y, predicao)
        trial.set_user_attr("r2_loo", r2_score(y, predicao))
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=SEED)
    )
    estudo.optimize(objetivo, n_trials=25, show_progress_bar=False)
    melhores = estudo.best_params
    n_bandas = int(melhores["n_bandas_ipls"])
    n_componentes = int(melhores["n_componentes"])
    bandas_finais = escolhidas.head(n_bandas)["banda_nm"].astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)
    pred_loo = predizer_loo(x_final, y, n_componentes)
    pred_ajuste = PLSRegression(n_components=n_componentes, scale=True).fit(x_final, y).predict(x_final).ravel()

    # Referência: configuração produzida pelo pipeline-base antes da otimização.
    bandas_base = escolhidas["banda_nm"].astype(int).tolist()
    x_base = dados[[str(banda) for banda in bandas_base]].to_numpy(float)
    pred_base = predizer_loo(x_base, y, 5)
    comparacao = pd.DataFrame([
        {"modelo": "PLSR base", "n_bandas": 5, "n_componentes": 5,
         "bandas_nm": ", ".join(map(str, bandas_base)), "r2_loo": r2_score(y, pred_base),
         "rmse_loo": root_mean_squared_error(y, pred_base)},
        {"modelo": "PLSR otimizado por Optuna", "n_bandas": n_bandas, "n_componentes": n_componentes,
         "bandas_nm": ", ".join(map(str, bandas_finais)), "r2_loo": r2_score(y, pred_loo),
         "rmse_loo": root_mean_squared_error(y, pred_loo)},
    ])
    comparacao["delta_r2_vs_base"] = comparacao["r2_loo"] - comparacao.loc[0, "r2_loo"]
    comparacao["delta_rmse_vs_base"] = comparacao["rmse_loo"] - comparacao.loc[0, "rmse_loo"]
    comparacao.to_csv(SAIDA / "comparacao_metricas_loo.csv", sep=";", index=False)
    estudo.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(
        SAIDA / "trials_optuna.csv", sep=";", index=False
    )
    pd.DataFrame([{
        "n_trials": len(estudo.trials), "seed": SEED, "p_limiar_spearman": P_LIMIAR,
        "largura_janela_ipls_nm": LARGURA_JANELA_NM, "n_bandas_ipls": n_bandas,
        "n_componentes": n_componentes, "bandas_nm": ", ".join(map(str, bandas_finais)),
        "r2_ajuste": r2_score(y, pred_ajuste), "rmse_ajuste": root_mean_squared_error(y, pred_ajuste),
        "r2_loo": r2_score(y, pred_loo), "rmse_loo": root_mean_squared_error(y, pred_loo),
    }]).to_csv(SAIDA / "melhor_configuracao_optuna.csv", sep=";", index=False)
    dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].assign(
        chltotal_predito_loo=pred_loo, chltotal_predito_ajuste=pred_ajuste
    ).to_csv(SAIDA / "predicoes_melhor_modelo_optuna.csv", sep=";", index=False)
    print(comparacao.to_string(index=False))


if __name__ == "__main__":
    main()
