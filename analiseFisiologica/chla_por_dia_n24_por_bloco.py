#!/usr/bin/env python3
"""PLSR de Clorofila A por data, com validação LOO e gráficos por condição."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error

import clorofila_total_spearman_plsr as base
import cra_interval_pls_vip as ipls
from gs_por_dia_n24_por_bloco import BANDAS, DATAS, prever


ROOT = Path(__file__).resolve().parents[1]
SAIDA = Path(__file__).resolve().parent / "resultados_chla_por_dia_n24_por_bloco"
CORES = {"IRR": "#2474b5", "NIR": "#d95f02"}
ROTULOS = {"IRR": "Irrigado", "NIR": "Não irrigado"}


def carregar(dia, coleta):
    fis = pd.read_excel(ROOT / "dataset" / "FisiologicosEduarda.xlsx")
    fis = fis.rename(columns=lambda coluna: str(coluna).replace(".", "/"))
    fis = fis.rename(columns={"Genótipo": "genotipo", "Condição": "condicao", f"CHL A ({dia})": "chla"})
    fis["condicao"] = fis["condicao"].map({"IRR": "IRR", "NIR": "NIR"})
    fis["chla"] = pd.to_numeric(fis["chla"], errors="coerce")
    fis["ordem_repeticao_fisiologia"] = fis.groupby(["genotipo", "condicao"], sort=False).cumcount() + 1
    fis["bloco"] = "B" + fis["ordem_repeticao_fisiologia"].astype(str)
    fis = fis.dropna(subset=["chla"])
    esp = pd.read_csv(ROOT / "dataset" / "Unificada13052026_Limpa.csv", sep=";")
    esp = esp.loc[esp.data_coleta.eq(coleta) & esp.turno.eq("manha")].copy()
    esp["condicao"] = esp["condicao"].map({"IRRIG": "IRR", "NIRRIG": "NIR"})
    for banda in BANDAS:
        esp[banda] = pd.to_numeric(esp[banda].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    esp = esp.groupby(["bloco", "genotipo", "condicao"], as_index=False)[BANDAS].mean()
    dados = fis.merge(esp, on=["bloco", "genotipo", "condicao"], validate="one_to_one")
    if len(dados) != 24:
        raise ValueError(f"{dia}: esperavam-se 24 pares; encontrados {len(dados)}.")
    return dados


def plotar(dados, loo, dia, r2, rmse, arquivo):
    y = dados.chla
    minimo, maximo = min(y.min(), loo.min()), max(y.max(), loo.max())
    margem = (maximo - minimo) * .06
    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    for condicao in ("IRR", "NIR"):
        grupo = dados.condicao.eq(condicao)
        ax.scatter(y[grupo], loo[grupo], s=66, color=CORES[condicao], edgecolor="white",
                   linewidth=.8, alpha=.92, label=ROTULOS[condicao])
    ax.plot([minimo - margem, maximo + margem], [minimo - margem, maximo + margem], "--", color="#333333", linewidth=1.2, label="1:1")
    ax.text(.03, .97, f"R² LOO = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#666", "alpha": .94})
    ax.set(xlim=(minimo - margem, maximo + margem), ylim=(minimo - margem, maximo + margem),
           xlabel="Clorofila A observada", ylabel="Clorofila A predita (LOO)",
           title=f"Clorofila A: observado × predito — {dia}")
    ax.grid(alpha=.22); ax.legend(title="Condição", frameon=True, loc="lower right")
    fig.tight_layout(); fig.savefig(arquivo.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(arquivo.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def executar(nome, dia, coleta):
    dados = carregar(dia, coleta); y = dados.chla.to_numpy(float); x = dados[BANDAS].to_numpy(float)
    corr = [base.spearman_com_p_exato(x[:, i], y) for i in range(x.shape[1])]
    tabela = pd.DataFrame({"banda_nm": np.asarray(BANDAS, dtype=int), "rho_spearman": [item[0] for item in corr], "p_valor": [item[1] for item in corr]})
    retidas = tabela.loc[tabela.p_valor < .05, "banda_nm"].astype(str).tolist()
    # Em dias com poucas bandas elegíveis, usa o maior conjunto não colinear
    # disponível, em vez de forçar cinco bandas sem suporte estatístico.
    for quantidade in range(5, 0, -1):
        try:
            escolhidas = ipls.selecionar_ipls_diverso(
                dados, retidas, y, tabela, largura_intervalo_nm=10,
                limiar_correlacao=None, n_bandas=quantidade
            ).reset_index(drop=True)
            break
        except ValueError:
            if quantidade == 1:
                raise
    bandas = escolhidas.banda_nm.astype(int).tolist(); x_final = dados[[str(banda) for banda in bandas]].to_numpy(float)
    espaco = {"n_componentes": list(range(1, min(5, len(bandas)) + 1)), "scale": [True, False]}
    def objetivo(trial):
        return root_mean_squared_error(y, prever(x_final, y, int(trial.suggest_categorical("n_componentes", espaco["n_componentes"])), bool(trial.suggest_categorical("scale", espaco["scale"]))))
    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(espaco, seed=20260929)); estudo.optimize(objetivo, n_trials=10, show_progress_bar=False)
    n, scale = int(estudo.best_params["n_componentes"]), bool(estudo.best_params["scale"]); loo = prever(x_final, y, n, scale)
    modelo = PLSRegression(n_components=n, scale=scale).fit(x_final, y); ajuste = modelo.predict(x_final).ravel(); r2, rmse = r2_score(y, loo), root_mean_squared_error(y, loo)
    escolhidas["vip_plsr_final"] = base.calcular_vip(modelo); escolhidas.sort_values("vip_plsr_final", ascending=False).to_csv(SAIDA / f"{nome}_top5_vip.csv", sep=";", index=False)
    dados[["bloco", "genotipo", "condicao", "ordem_repeticao_fisiologia", "chla"]].assign(predito_ajuste=ajuste, predito_loo=loo).to_csv(SAIDA / f"{nome}_predicoes.csv", sep=";", index=False)
    plotar(dados, loo, dia, r2, rmse, SAIDA / f"chla_observado_x_predito_{nome}")
    return {"data": dia, "n": len(dados), "bandas_top5_nm": ", ".join(map(str, bandas)), "n_componentes": n, "scale": scale, "r2_loo": r2, "rmse_loo": rmse}


def consolidar():
    resumo = []
    for nome, (dia, _) in DATAS.items():
        predicoes = pd.read_csv(SAIDA / f"{nome}_predicoes.csv", sep=";")
        bandas = pd.read_csv(SAIDA / f"{nome}_top5_vip.csv", sep=";")
        resumo.append({"data": dia, "n": len(predicoes),
                       "bandas_vip_nm": ", ".join(map(str, bandas.banda_nm.astype(int).tolist())),
                       "r2_loo": r2_score(predicoes.chla, predicoes.predito_loo),
                       "rmse_loo": root_mean_squared_error(predicoes.chla, predicoes.predito_loo)})
    pd.DataFrame(resumo).to_csv(SAIDA / "resumo_chla_por_dia.csv", sep=";", index=False, float_format="%.6f")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING); SAIDA.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "consolidar":
        consolidar()
        return
    nomes = sys.argv[1:] if len(sys.argv) > 1 else list(DATAS)
    resumo = [executar(nome, *DATAS[nome]) for nome in nomes]
    pd.DataFrame(resumo).to_csv(SAIDA / "resumo_chla_por_dia.csv", sep=";", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
