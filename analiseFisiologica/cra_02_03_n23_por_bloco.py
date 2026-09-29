#!/usr/bin/env python3
"""CRA 02/03: PLSR por bloco com as 23 leituras fisiológicas disponíveis."""

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
SAIDA = Path(__file__).resolve().parent / "resultados_cra_02_03_n23_por_bloco"
BANDAS = [str(i) for i in range(350, 2501)]
ESTILOS = {"IRR": {"marker": "s", "color": "#2563eb", "label": "Irrigado"},
           "NIR": {"marker": "^", "color": "#dc2626", "label": "Não irrigado"}}


def loo(x, y, n, scale):
    return cross_val_predict(PLSRegression(n_components=n, scale=scale), x, y, cv=LeaveOneOut()).ravel()


def carregar():
    fis = pd.read_excel(ROOT / "dataset" / "FisiologicosEduarda.xlsx")
    fis = fis.rename(columns=lambda coluna: str(coluna).replace(".", "/"))
    fis = fis.rename(columns={"Genótipo": "genotipo", "Condição": "condicao", "CRA (02/03)": "cra"})
    fis["condicao"] = fis["condicao"].map({"IRR": "IRR", "NIR": "NIR"})
    fis["cra"] = pd.to_numeric(fis["cra"], errors="coerce")
    # O bloco é inferido pela posição original, antes de remover a CRA ausente.
    fis["ordem_repeticao_fisiologia"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount() + 1
    fis["bloco"] = "B" + fis["ordem_repeticao_fisiologia"].astype(str)
    fis = fis.dropna(subset=["cra"]).copy()
    esp = pd.read_csv(ROOT / "dataset" / "Unificada13052026_Limpa.csv", sep=";")
    esp = esp.loc[esp["data_coleta"].eq("D09M") & esp["turno"].eq("manha")].copy()
    esp["condicao"] = esp["condicao"].map({"IRRIG": "IRR", "NIRRIG": "NIR"})
    for banda in BANDAS:
        esp[banda] = pd.to_numeric(esp[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    esp = esp.groupby(["bloco", "genotipo", "condicao"], as_index=False)[BANDAS].mean()
    dados = fis.merge(esp, on=["bloco", "genotipo", "condicao"], validate="one_to_one")
    if len(dados) != 23:
        raise ValueError(f"Esperavam-se 23 pares válidos; encontrados {len(dados)}.")
    return dados


def plotar(dados, y, ajuste, pred, r2, rmse):
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    for eixo, previsto, titulo in ((eixos[0], ajuste, "Ajuste"), (eixos[1], pred, "Validação leave-one-out")):
        for condicao, estilo in ESTILOS.items():
            mascara = dados["condicao"].eq(condicao)
            eixo.scatter(y[mascara], previsto[mascara], s=55, edgecolor="white", linewidth=.6, **estilo)
        baixo, alto = min(y.min(), previsto.min()) - .03, max(y.max(), previsto.max()) + .03
        eixo.plot([baixo, alto], [baixo, alto], "--", color="#475569", linewidth=1)
        eixo.set(xlim=(baixo, alto), ylim=(baixo, alto), xlabel="CRA observado", ylabel="CRA predito", title=titulo)
    eixos[0].legend(frameon=False)
    eixos[1].text(.04, .95, f"n = 23\nR² LOO = {r2:.3f}\nRMSE LOO = {rmse:.3f}", transform=eixos[1].transAxes,
                  va="top", bbox={"facecolor": "white", "edgecolor": "#94a3b8", "boxstyle": "round,pad=.3"})
    fig.suptitle("CRA — 02/03: PLSR por bloco (n = 23)")
    fig.savefig(SAIDA / "observado_vs_predito_n23.png", dpi=220)
    plt.close(fig)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    SAIDA.mkdir(exist_ok=True)
    dados = carregar()
    y, x = dados["cra"].to_numpy(float), dados[BANDAS].to_numpy(float)
    corr = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.asarray(BANDAS, dtype=int), "rho_spearman": [v[0] for v in corr], "p_valor": [v[1] for v in corr]})
    retidas = tabela.loc[tabela["p_valor"] < .05, "banda_nm"].astype(str).tolist()
    escolhidas = ipls.selecionar_ipls_diverso(dados, retidas, y, tabela, largura_intervalo_nm=10, limiar_correlacao=None, n_bandas=5).reset_index(drop=True)
    bandas = escolhidas["banda_nm"].astype(int).tolist()
    xf = dados[[str(banda) for banda in bandas]].to_numpy(float)
    espaco = {"n_componentes": [1, 2, 3, 4, 5], "scale": [True, False]}
    def objetivo(trial):
        pred = loo(xf, y, int(trial.suggest_categorical("n_componentes", espaco["n_componentes"])), bool(trial.suggest_categorical("scale", espaco["scale"])))
        trial.set_user_attr("r2_loo", r2_score(y, pred))
        return root_mean_squared_error(y, pred)
    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=20260928))
    estudo.optimize(objetivo, n_trials=10, show_progress_bar=False)
    n, scale = int(estudo.best_params["n_componentes"]), bool(estudo.best_params["scale"])
    pred = loo(xf, y, n, scale)
    modelo = PLSRegression(n_components=n, scale=scale).fit(xf, y)
    ajuste = modelo.predict(xf).ravel()
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
    escolhidas.sort_values("vip_plsr_final", ascending=False).to_csv(SAIDA / "top5_vip.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", "ordem_repeticao_fisiologia", "cra"]].assign(predito_ajuste=ajuste, predito_loo=pred).to_csv(SAIDA / "dados_pareados_n23_predicoes.csv", sep=";", index=False)
    estudo.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(SAIDA / "trials_optuna.csv", sep=";", index=False)
    r2, rmse = r2_score(y, pred), root_mean_squared_error(y, pred)
    pd.DataFrame([{"n": 23, "observacao_excluida": "BR16, IRR, B1: CRA ausente", "p_limiar_spearman": .05, "n_bandas_spearman": len(retidas), "bandas_nm": ", ".join(map(str, bandas)), "n_componentes": n, "scale": scale, "r2_ajuste": r2_score(y, ajuste), "rmse_ajuste": root_mean_squared_error(y, ajuste), "r2_loo": r2, "rmse_loo": rmse, "pareamento": "ordem original das repetições fisiológicas mapeada para B1-B4"}]).to_csv(SAIDA / "resumo_n23.csv", sep=";", index=False)
    plotar(dados, y, ajuste, pred, r2, rmse)


if __name__ == "__main__":
    main()
