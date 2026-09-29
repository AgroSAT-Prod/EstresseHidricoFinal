#!/usr/bin/env python3
"""Compara limiares Spearman para CHL Total com todas as datas agrupadas."""

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


SAIDA = Path(__file__).resolve().parent / "resultados_chltotal_todas_datas_limiares_optuna"
LIMIARES = (0.001, 0.05, 0.15)
SEED = 20260928
ESTILOS = {"IRR": {"marker": "s", "color": "#2563eb", "label": "Irrigado"},
           "NIR": {"marker": "^", "color": "#dc2626", "label": "Não irrigado"}}


def rotulo(limiar: float) -> str:
    return f"p{limiar:.3f}".replace(".", "_")


def prever(x: np.ndarray, y: np.ndarray, componentes: int, scale: bool) -> np.ndarray:
    return cross_val_predict(PLSRegression(n_components=componentes, scale=scale), x, y, cv=LeaveOneOut()).ravel()


def plotar(dados: pd.DataFrame, y: np.ndarray, pred: np.ndarray, limiar: float, r2: float, rmse: float) -> None:
    fig, eixo = plt.subplots(figsize=(5.8, 5.1), layout="constrained")
    for condicao, estilo in ESTILOS.items():
        mascara = dados["condicao"].eq(condicao)
        eixo.scatter(y[mascara], pred[mascara], s=50, edgecolor="white", linewidth=.6, **estilo)
    minimo, maximo = min(y.min(), pred.min()) - .5, max(y.max(), pred.max()) + .5
    eixo.plot([minimo, maximo], [minimo, maximo], "--", color="#475569", linewidth=1)
    eixo.set(xlim=(minimo, maximo), ylim=(minimo, maximo), xlabel="CHL Total observado", ylabel="CHL Total predito",
              title=f"Todas as datas — Spearman p < {limiar:g}")
    eixo.legend(frameon=False)
    eixo.text(.04, .95, f"R² LOO = {r2:.3f}\nRMSE LOO = {rmse:.3f}", transform=eixo.transAxes, va="top",
              bbox={"facecolor": "white", "edgecolor": "#94a3b8", "boxstyle": "round,pad=.3"})
    fig.savefig(SAIDA / f"{rotulo(limiar)}_observado_vs_predito.png", dpi=220)
    plt.close(fig)


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    SAIDA.mkdir(exist_ok=True)
    fisio = base.ler_fisiologia()
    espectros, bandas = base.ler_espectros()
    dados = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    y = dados["chl_total"].to_numpy(float)
    x_todas = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x_todas[:, indice], y) for indice in range(x_todas.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.asarray(bandas, dtype=int), "rho_spearman": [r[0] for r in correlacoes],
                           "p_valor": [r[1] for r in correlacoes]})
    tabela.to_csv(SAIDA / "spearman_todas_bandas.csv", sep=";", index=False)
    resumo = []
    for limiar in LIMIARES:
        prefixo = rotulo(limiar)
        retidas = tabela.loc[tabela["p_valor"] < limiar, "banda_nm"].astype(str).tolist()
        escolhidas = ipls.selecionar_ipls_diverso(dados, retidas, y, tabela, largura_intervalo_nm=10,
                                                  limiar_correlacao=None, n_bandas=5).reset_index(drop=True)
        bandas_finais = escolhidas["banda_nm"].astype(int).tolist()
        x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)
        espaco = {"n_componentes": [1, 2, 3, 4, 5], "scale": [True, False]}
        def objetivo(trial: optuna.Trial) -> float:
            n = int(trial.suggest_categorical("n_componentes", espaco["n_componentes"]))
            scale = bool(trial.suggest_categorical("scale", espaco["scale"]))
            predicao = prever(x_final, y, n, scale)
            trial.set_user_attr("r2_loo", r2_score(y, predicao))
            return root_mean_squared_error(y, predicao)
        estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=SEED))
        estudo.optimize(objetivo, n_trials=10, show_progress_bar=False)
        n, scale = int(estudo.best_params["n_componentes"]), bool(estudo.best_params["scale"])
        predicao = prever(x_final, y, n, scale)
        modelo = PLSRegression(n_components=n, scale=scale).fit(x_final, y)
        escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
        escolhidas.sort_values("vip_plsr_final", ascending=False).to_csv(SAIDA / f"{prefixo}_top5_vip.csv", sep=";", index=False)
        dados[["data_fisiologica", "genotipo", "condicao", "chl_total"]].assign(predito_loo=predicao).to_csv(
            SAIDA / f"{prefixo}_predicoes_loo.csv", sep=";", index=False)
        estudo.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(
            SAIDA / f"{prefixo}_trials_optuna.csv", sep=";", index=False)
        r2, rmse = r2_score(y, predicao), root_mean_squared_error(y, predicao)
        plotar(dados, y, predicao, limiar, r2, rmse)
        resumo.append({"p_limiar_spearman": limiar, "n_grupos": len(dados), "n_bandas_spearman": len(retidas),
                       "bandas_nm": ", ".join(map(str, bandas_finais)), "n_componentes_optuna": n,
                       "scale_optuna": scale, "r2_loo": r2, "rmse_loo": rmse})
    pd.DataFrame(resumo).to_csv(SAIDA / "comparacao_limiares_todas_datas.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
