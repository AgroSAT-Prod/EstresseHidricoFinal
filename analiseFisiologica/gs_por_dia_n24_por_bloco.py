#!/usr/bin/env python3
"""PLSR de gs por data, com validação leave-one-out e gráficos por condição."""
from pathlib import Path
import sys

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
SAIDA = Path(__file__).resolve().parent / "resultados_gs_por_dia_n24_por_bloco"
BANDAS = [str(i) for i in range(350, 2501)]
DATAS = {
    "23_02": ("23/02", "D02M"), "24_02": ("24/02", "D03M"),
    "25_02": ("25/02", "D04M"), "26_02": ("26/02", "D05M"),
    "27_02": ("27/02", "D06M"), "02_03": ("02/03", "D09M"),
    "03_03": ("03/03", "D10M"),
}
CORES = {"IRR": "#2474b5", "NIR": "#d95f02"}
ROTULOS = {"IRR": "Irrigado", "NIR": "Não irrigado"}


def prever(x, y, n_componentes, scale):
    return cross_val_predict(PLSRegression(n_components=n_componentes, scale=scale), x, y,
                             cv=LeaveOneOut()).ravel()


def carregar(dia, coleta):
    fis = pd.read_excel(ROOT / "dataset" / "FisiologicosEduarda.xlsx")
    fis = fis.rename(columns=lambda c: str(c).replace(".", "/"))
    fis = fis.rename(columns={"Genótipo": "genotipo", "Condição": "condicao", f"gs ({dia})": "gs"})
    fis["condicao"] = fis["condicao"].map({"IRR": "IRR", "NIR": "NIR"})
    fis["gs"] = pd.to_numeric(fis["gs"], errors="coerce")
    fis["ordem_repeticao_fisiologia"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount() + 1
    fis["bloco"] = "B" + fis["ordem_repeticao_fisiologia"].astype(str)
    fis = fis.dropna(subset=["gs"])

    esp = pd.read_csv(ROOT / "dataset" / "Unificada13052026_Limpa.csv", sep=";")
    esp = esp.loc[esp["data_coleta"].eq(coleta) & esp["turno"].eq("manha")].copy()
    esp["condicao"] = esp["condicao"].map({"IRRIG": "IRR", "NIRRIG": "NIR"})
    for banda in BANDAS:
        esp[banda] = pd.to_numeric(esp[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    esp = esp.groupby(["bloco", "genotipo", "condicao"], as_index=False)[BANDAS].mean()
    dados = fis.merge(esp, on=["bloco", "genotipo", "condicao"], validate="one_to_one")
    if len(dados) != 24:
        raise ValueError(f"{dia}: esperavam-se 24 pares; encontrados {len(dados)}.")
    return dados


def plotar(dados, predito, dia, r2, rmse, arquivo):
    y = dados["gs"]
    minimo, maximo = min(y.min(), predito.min()), max(y.max(), predito.max())
    margem = (maximo - minimo) * .06
    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    for condicao in ("IRR", "NIR"):
        grupo = dados.condicao.eq(condicao)
        ax.scatter(y[grupo], predito[grupo], s=66, color=CORES[condicao], edgecolor="white",
                   linewidth=.8, alpha=.92, label=ROTULOS[condicao])
    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem], "--",
            color="#333333", linewidth=1.2, label="1:1")
    ax.text(.03, .97, f"R² LOO = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
            va="top", ha="left", bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
            "edgecolor": "#666", "alpha": .94})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel="gs observada", ylabel="gs predita (LOO)",
           title=f"Condutância estomática: observado × predito — {dia}")
    ax.grid(alpha=.22); ax.legend(title="Condição", frameon=True, loc="lower right")
    fig.tight_layout(); fig.savefig(arquivo.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(arquivo.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def executar_data(nome, dia, coleta):
    dados = carregar(dia, coleta)
    y, x = dados.gs.to_numpy(float), dados[BANDAS].to_numpy(float)
    correlacoes = [base.spearman_com_p_exato(x[:, indice], y) for indice in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.asarray(BANDAS, dtype=int),
                           "rho_spearman": [item[0] for item in correlacoes],
                           "p_valor": [item[1] for item in correlacoes]})
    retidas = tabela.loc[tabela.p_valor < .05, "banda_nm"].astype(str).tolist()
    escolhidas = ipls.selecionar_ipls_diverso(dados, retidas, y, tabela, largura_intervalo_nm=10,
                                               limiar_correlacao=None, n_bandas=5).reset_index(drop=True)
    bandas = escolhidas.banda_nm.astype(int).tolist()
    x_final = dados[[str(banda) for banda in bandas]].to_numpy(float)
    espaco = {"n_componentes": [1, 2, 3, 4, 5], "scale": [True, False]}

    def objetivo(trial):
        n = int(trial.suggest_categorical("n_componentes", espaco["n_componentes"]))
        scale = bool(trial.suggest_categorical("scale", espaco["scale"]))
        return root_mean_squared_error(y, prever(x_final, y, n, scale))

    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=20260929))
    estudo.optimize(objetivo, n_trials=10, show_progress_bar=False)
    n, scale = int(estudo.best_params["n_componentes"]), bool(estudo.best_params["scale"])
    loo = prever(x_final, y, n, scale)
    modelo = PLSRegression(n_components=n, scale=scale).fit(x_final, y)
    ajuste = modelo.predict(x_final).ravel()
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo)
    escolhidas.sort_values("vip_plsr_final", ascending=False).to_csv(SAIDA / f"{nome}_top5_vip.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", "ordem_repeticao_fisiologia", "gs"]].assign(
        predito_ajuste=ajuste, predito_loo=loo).to_csv(SAIDA / f"{nome}_predicoes.csv", sep=";", index=False)
    r2, rmse = r2_score(y, loo), root_mean_squared_error(y, loo)
    plotar(dados, loo, dia, r2, rmse, SAIDA / f"gs_observado_x_predito_{nome}")
    return {"data": dia, "n": len(dados), "bandas_top5_nm": ", ".join(map(str, bandas)),
            "n_componentes": n, "scale": scale, "r2_loo": r2, "rmse_loo": rmse}


def consolidar_resultados():
    resumo = []
    for nome, (dia, _) in DATAS.items():
        predicoes = pd.read_csv(SAIDA / f"{nome}_predicoes.csv", sep=";")
        bandas = pd.read_csv(SAIDA / f"{nome}_top5_vip.csv", sep=";")
        resumo.append({
            "data": dia, "n": len(predicoes),
            "bandas_top5_nm": ", ".join(map(str, bandas.banda_nm.astype(int).tolist())),
            "r2_loo": r2_score(predicoes.gs, predicoes.predito_loo),
            "rmse_loo": root_mean_squared_error(predicoes.gs, predicoes.predito_loo),
        })
    pd.DataFrame(resumo).to_csv(SAIDA / "resumo_gs_por_dia.csv", sep=";", index=False,
                                float_format="%.6f")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    SAIDA.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "consolidar":
        consolidar_resultados()
        return
    datas = DATAS
    if len(sys.argv) > 1:
        nome = sys.argv[1]
        if nome not in DATAS:
            raise ValueError(f"Data inválida: {nome}")
        datas = {nome: DATAS[nome]}
    resumo = [executar_data(nome, dia, coleta) for nome, (dia, coleta) in datas.items()]
    pd.DataFrame(resumo).to_csv(SAIDA / "resumo_gs_por_dia.csv", sep=";", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
