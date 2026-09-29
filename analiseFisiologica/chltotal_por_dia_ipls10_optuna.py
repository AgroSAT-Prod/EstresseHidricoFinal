#!/usr/bin/env python3
"""CHL Total por dia: Spearman p < 0,05, iPLS 10 nm e Optuna-PLSR.

As análises usam as médias genótipo × condição de cada data (n=6). Nos
gráficos de dispersão, IRRIG é mostrado como quadrado azul e NIRRIG como
triângulo vermelho.
"""

from __future__ import annotations

import argparse
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


SAIDA = Path(__file__).resolve().parent / "resultados_chltotal_por_dia_p005_ipls10_optuna"
P_LIMIAR = .05
LARGURA_JANELA_NM = 10
SEED = 20260928
ESTILOS = {
    "IRR": {"marker": "s", "color": "#2563eb", "label": "Irrigado"},
    "NIR": {"marker": "^", "color": "#dc2626", "label": "Não irrigado"},
}


def nome_data(data: str) -> str:
    return data.replace("/", "_")


def prever_loo(x: np.ndarray, y: np.ndarray, n_componentes: int, scale: bool) -> np.ndarray:
    return cross_val_predict(
        PLSRegression(n_components=n_componentes, scale=scale), x, y, cv=LeaveOneOut()
    ).ravel()


def dispersao_predicoes(dados: pd.DataFrame, y: np.ndarray, ajuste: np.ndarray, loo: np.ndarray,
                         data: str, r2_loo: float, rmse_loo: float) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    for eixo, predito, titulo in ((eixos[0], ajuste, "Ajuste"), (eixos[1], loo, "Validação leave-one-out")):
        for condicao, estilo in ESTILOS.items():
            mascara = dados["condicao"].eq(condicao)
            eixo.scatter(y[mascara], predito[mascara], s=65, edgecolor="white", linewidth=.7, **estilo)
        limite_inf = min(y.min(), predito.min()) - .4
        limite_sup = max(y.max(), predito.max()) + .4
        eixo.plot([limite_inf, limite_sup], [limite_inf, limite_sup], "--", color="#475569", linewidth=1)
        eixo.set(xlim=(limite_inf, limite_sup), ylim=(limite_inf, limite_sup), xlabel="CHL Total observado",
                  ylabel="CHL Total predito", title=titulo)
    eixos[0].legend(frameon=False)
    eixos[1].text(.04, .95, f"R² LOO = {r2_loo:.3f}\nRMSE LOO = {rmse_loo:.2f}", transform=eixos[1].transAxes,
                  va="top", bbox={"facecolor": "white", "edgecolor": "#94a3b8", "boxstyle": "round,pad=.3"})
    fig.suptitle(f"CHL Total — {data}: PLSR otimizado por Optuna")
    fig.savefig(SAIDA / f"{nome_data(data)}_observado_vs_predito.png", dpi=220)
    plt.close(fig)


def dispersao_top5(dados: pd.DataFrame, top5: pd.DataFrame, data: str) -> None:
    fig, eixos = plt.subplots(1, len(top5), figsize=(3.55 * len(top5), 4.2), layout="constrained")
    for eixo, (_, banda) in zip(np.atleast_1d(eixos), top5.iterrows()):
        coluna = str(int(banda["banda_nm"]))
        x, y = dados[coluna].to_numpy(float), dados["chl_total"].to_numpy(float)
        coeficiente, intercepto = np.polyfit(x, y, 1)
        grade = np.linspace(x.min(), x.max(), 100)
        for condicao, estilo in ESTILOS.items():
            mascara = dados["condicao"].eq(condicao)
            eixo.scatter(x[mascara], y[mascara], s=65, edgecolor="white", linewidth=.7, **estilo)
        eixo.plot(grade, coeficiente * grade + intercepto, "--", color="#475569", linewidth=1)
        eixo.set(xlabel="Reflectância", ylabel="CHL Total",
                  title=f"#{int(banda['rank_vip'])}: {coluna} nm\nVIP={banda['vip_plsr_final']:.2f}; R²={banda['r2_linear']:.2f}")
    np.atleast_1d(eixos)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{data}: reflectância × CHL Total — bandas iPLS/VIP")
    fig.savefig(SAIDA / f"{nome_data(data)}_reflectancia_x_chltotal_top5.png", dpi=220)
    plt.close(fig)


def analisar_data(data: str, dados: pd.DataFrame, bandas: list[str]) -> dict[str, object]:
    y = dados["chl_total"].to_numpy(float)
    x_todas = dados[bandas].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x_todas[:, i], y) for i in range(x_todas.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.asarray(bandas, dtype=int), "rho_spearman": [v[0] for v in correlacoes],
                           "p_valor": [v[1] for v in correlacoes]})
    retidas = tabela.loc[tabela["p_valor"] < P_LIMIAR, "banda_nm"].astype(str).tolist()
    prefixo = nome_data(data)
    tabela.to_csv(SAIDA / f"{prefixo}_spearman.csv", sep=";", index=False)
    registro: dict[str, object] = {"data": data, "n_grupos": len(dados), "n_bandas_spearman": len(retidas),
                                   "p_limiar_spearman": P_LIMIAR}
    # Com apenas seis grupos por dia, a ausência de associação Spearman não
    # impede uma análise exploratória. Nesse caso o iPLS recebe todo o espectro.
    # A estratégia fica registrada para não confundir esse modelo com os que
    # efetivamente passaram pelo filtro p < 0,05.
    candidatas = retidas if retidas else bandas
    estrategia = (f"Spearman p < {P_LIMIAR:g} + iPLS" if retidas
                  else f"iPLS em todas as bandas (exploratório; sem Spearman p < {P_LIMIAR:g})")
    try:
        escolhidas = ipls.selecionar_ipls_diverso(dados, candidatas, y, tabela, largura_intervalo_nm=LARGURA_JANELA_NM,
                                                  limiar_correlacao=None, n_bandas=5).reset_index(drop=True)
    except ValueError as erro:
        registro["status"] = str(erro)
        return registro
    bandas_finais = escolhidas["banda_nm"].astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)
    max_componentes = min(4, x_final.shape[1])  # n-2 no treino de cada dobra LOO
    espaco = {"n_componentes": list(range(1, max_componentes + 1)), "scale": [True, False]}
    def objetivo(trial: optuna.Trial) -> float:
        n = int(trial.suggest_categorical("n_componentes", espaco["n_componentes"]))
        scale = bool(trial.suggest_categorical("scale", espaco["scale"]))
        pred = prever_loo(x_final, y, n, scale)
        trial.set_user_attr("r2_loo", r2_score(y, pred))
        return root_mean_squared_error(y, pred)
    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=SEED))
    estudo.optimize(objetivo, n_trials=max_componentes * 2, show_progress_bar=False)
    n = int(estudo.best_params["n_componentes"])
    scale = bool(estudo.best_params["scale"])
    loo = prever_loo(x_final, y, n, scale)
    modelo = PLSRegression(n_components=n, scale=scale).fit(x_final, y)
    ajuste = modelo.predict(x_final).ravel()
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
    top5 = escolhidas.sort_values("vip_plsr_final", ascending=False).reset_index(drop=True)
    top5.insert(0, "rank_vip", range(1, len(top5) + 1))
    r2_lineares, rmse_lineares = [], []
    for banda in top5["banda_nm"].astype(int):
        valor = dados[str(banda)].to_numpy(float)
        predito = np.polyval(np.polyfit(valor, y, 1), valor)
        r2_lineares.append(r2_score(y, predito))
        rmse_lineares.append(root_mean_squared_error(y, predito))
    top5["r2_linear"], top5["rmse_linear"] = r2_lineares, rmse_lineares
    top5.to_csv(SAIDA / f"{prefixo}_top5_vip.csv", sep=";", index=False)
    dados[["genotipo", "condicao", "chl_total"]].assign(predito_ajuste=ajuste, predito_loo=loo).to_csv(
        SAIDA / f"{prefixo}_predicoes.csv", sep=";", index=False)
    estudo.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(
        SAIDA / f"{prefixo}_trials_optuna.csv", sep=";", index=False)
    dispersao_predicoes(dados, y, ajuste, loo, data, r2_score(y, loo), root_mean_squared_error(y, loo))
    dispersao_top5(dados, top5, data)
    registro.update({"status": "Modelo ajustado", "estrategia_selecao": estrategia,
                     "n_bandas_finais": len(top5), "n_componentes": n, "scale": scale,
                     "bandas_nm": ", ".join(map(str, bandas_finais)), "r2_ajuste": r2_score(y, ajuste),
                     "rmse_ajuste": root_mean_squared_error(y, ajuste), "r2_loo": r2_score(y, loo),
                     "rmse_loo": root_mean_squared_error(y, loo)})
    return registro


def main() -> None:
    global P_LIMIAR, SAIDA
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-limiar", type=float, default=P_LIMIAR)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    if not 0 < args.p_limiar < 1:
        raise ValueError("--p-limiar deve estar entre 0 e 1.")
    P_LIMIAR, SAIDA = args.p_limiar, args.saida
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    SAIDA.mkdir(exist_ok=True)
    fisio = base.ler_fisiologia()
    espectros, bandas = base.ler_espectros()
    todos = espectros.merge(fisio, on=["data_fisiologica", "genotipo", "condicao"], validate="one_to_one")
    resumo = [analisar_data(data, dados.reset_index(drop=True), bandas)
              for data, dados in todos.groupby("data_fisiologica", sort=False)]
    pd.DataFrame(resumo).to_csv(SAIDA / "resumo_por_dia.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
