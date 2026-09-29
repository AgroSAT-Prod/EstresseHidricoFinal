#!/usr/bin/env python3
"""PLSR de CHL Total em 23/02 usando 24 pares bloco × tratamento.

O arquivo fisiológico não traz o bloco. As quatro linhas, em sua ordem dentro
de cada genótipo × condição, são mapeadas para B1--B4; esse pareamento é uma
inferência auditável e é salvo junto aos resultados.
"""

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


ROOT = Path(__file__).resolve().parents[1]
SAIDA = Path(__file__).resolve().parent / "resultados_chltotal_23_02_n24_por_bloco"
BANDAS = [str(i) for i in range(350, 2501)]
ESTILOS = {"IRR": {"marker": "s", "color": "#2563eb", "label": "Irrigado"},
           "NIR": {"marker": "^", "color": "#dc2626", "label": "Não irrigado"}}


def prever(x: np.ndarray, y: np.ndarray, n: int, scale: bool) -> np.ndarray:
    return cross_val_predict(PLSRegression(n_components=n, scale=scale), x, y, cv=LeaveOneOut()).ravel()


def carregar() -> pd.DataFrame:
    fis = pd.read_excel(ROOT / "dataset" / "FisiologicosEduarda.xlsx")
    fis = fis.rename(columns=lambda coluna: str(coluna).replace(".", "/"))
    alvo = "CHL TOTAL (23/02)"
    fis = fis.rename(columns={"Genótipo": "genotipo", "Condição": "condicao", alvo: "chl_total"})
    fis["condicao"] = fis["condicao"].map({"IRR": "IRR", "NIR": "NIR"})
    fis["chl_total"] = pd.to_numeric(fis["chl_total"], errors="coerce")
    fis = fis.dropna(subset=["chl_total"]).copy()
    fis["ordem_repeticao_fisiologia"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount() + 1
    fis["bloco"] = "B" + fis["ordem_repeticao_fisiologia"].astype(str)
    if not fis.groupby(["genotipo", "condicao"]).size().eq(4).all():
        raise ValueError("Esperavam-se quatro leituras fisiológicas por genótipo × condição.")

    esp = pd.read_csv(ROOT / "dataset" / "Unificada13052026_Limpa.csv", sep=";")
    esp = esp.loc[esp["data_coleta"].eq("D02M") & esp["turno"].eq("manha")].copy()
    esp["condicao"] = esp["condicao"].map({"IRRIG": "IRR", "NIRRIG": "NIR"})
    for banda in BANDAS:
        esp[banda] = pd.to_numeric(esp[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    # Cada bloco possui oito curvas, resumidas em uma curva representativa.
    esp = esp.groupby(["bloco", "genotipo", "condicao"], as_index=False)[BANDAS].mean()
    dados = fis.merge(esp, on=["bloco", "genotipo", "condicao"], validate="one_to_one")
    if len(dados) != 24:
        raise ValueError(f"Pareamento incompleto: esperavam-se 24 linhas, foram obtidas {len(dados)}.")
    return dados


def plotar(dados: pd.DataFrame, y: np.ndarray, ajuste: np.ndarray, loo: np.ndarray, r2: float, rmse: float) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    for eixo, pred, titulo in ((eixos[0], ajuste, "Ajuste"), (eixos[1], loo, "Validação leave-one-out")):
        for condicao, estilo in ESTILOS.items():
            mascara = dados["condicao"].eq(condicao)
            eixo.scatter(y[mascara], pred[mascara], s=55, edgecolor="white", linewidth=.6, **estilo)
        baixo, alto = min(y.min(), pred.min()) - .5, max(y.max(), pred.max()) + .5
        eixo.plot([baixo, alto], [baixo, alto], "--", color="#475569", linewidth=1)
        eixo.set(xlim=(baixo, alto), ylim=(baixo, alto), xlabel="CHL Total observado", ylabel="CHL Total predito", title=titulo)
    eixos[0].legend(frameon=False)
    eixos[1].text(.04, .95, f"n = 24\nR² LOO = {r2:.3f}\nRMSE LOO = {rmse:.3f}", transform=eixos[1].transAxes,
                  va="top", bbox={"facecolor": "white", "edgecolor": "#94a3b8", "boxstyle": "round,pad=.3"})
    fig.suptitle("CHL Total — 23/02: PLSR por bloco (n = 24)")
    fig.savefig(SAIDA / "observado_vs_predito_n24.png", dpi=220)
    plt.close(fig)


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    SAIDA.mkdir(exist_ok=True)
    dados = carregar()
    y, x = dados["chl_total"].to_numpy(float), dados[BANDAS].to_numpy(float)
    resultados = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.asarray(BANDAS, dtype=int), "rho_spearman": [r[0] for r in resultados],
                           "p_valor": [r[1] for r in resultados]})
    retidas = tabela.loc[tabela["p_valor"] < .05, "banda_nm"].astype(str).tolist()
    escolhidas = ipls.selecionar_ipls_diverso(dados, retidas, y, tabela, largura_intervalo_nm=10,
                                              limiar_correlacao=None, n_bandas=5).reset_index(drop=True)
    bandas_finais = escolhidas["banda_nm"].astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas_finais]].to_numpy(float)
    espaco = {"n_componentes": [1, 2, 3, 4, 5], "scale": [True, False]}
    def objetivo(trial: optuna.Trial) -> float:
        pred = prever(x_final, y, int(trial.suggest_categorical("n_componentes", espaco["n_componentes"])),
                      bool(trial.suggest_categorical("scale", espaco["scale"])))
        trial.set_user_attr("r2_loo", r2_score(y, pred))
        return root_mean_squared_error(y, pred)
    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=20260928))
    estudo.optimize(objetivo, n_trials=10, show_progress_bar=False)
    n, scale = int(estudo.best_params["n_componentes"]), bool(estudo.best_params["scale"])
    loo = prever(x_final, y, n, scale)
    modelo = PLSRegression(n_components=n, scale=scale).fit(x_final, y)
    ajuste = modelo.predict(x_final).ravel()
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
    escolhidas.sort_values("vip_plsr_final", ascending=False).to_csv(SAIDA / "top5_vip.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", "ordem_repeticao_fisiologia", "chl_total"]].assign(
        predito_ajuste=ajuste, predito_loo=loo
    ).to_csv(SAIDA / "dados_pareados_n24_predicoes.csv", sep=";", index=False)
    estudo.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(SAIDA / "trials_optuna.csv", sep=";", index=False)
    r2, rmse = r2_score(y, loo), root_mean_squared_error(y, loo)
    pd.DataFrame([{"n": len(dados), "p_limiar_spearman": .05, "n_bandas_spearman": len(retidas),
                   "bandas_nm": ", ".join(map(str, bandas_finais)), "n_componentes": n, "scale": scale,
                   "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": root_mean_squared_error(y, ajuste),
                   "r2_loo": r2, "rmse_loo": rmse,
                   "pareamento": "ordem das 4 repetições fisiológicas mapeada para B1-B4"}]).to_csv(SAIDA / "resumo_n24.csv", sep=";", index=False)
    plotar(dados, y, ajuste, loo, r2, rmse)


if __name__ == "__main__":
    main()
