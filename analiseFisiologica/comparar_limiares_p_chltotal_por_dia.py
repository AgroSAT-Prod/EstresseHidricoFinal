#!/usr/bin/env python3
"""Consolida os resultados diários de CHL Total para três limiares Spearman."""

from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parent
LIMIARES = {
    "p_lt_0_001": RAIZ / "resultados_chltotal_por_dia_p0001_ipls10_optuna" / "resumo_por_dia.csv",
    "p_lt_0_05": RAIZ / "resultados_chltotal_por_dia_p005_ipls10_optuna" / "resumo_por_dia.csv",
    "p_lt_0_15": RAIZ / "resultados_chltotal_por_dia_p015_ipls10_optuna" / "resumo_por_dia.csv",
}
SAIDA = RAIZ / "resultados_chltotal_por_dia_comparacao_p"


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    tabelas = []
    for rotulo, arquivo in LIMIARES.items():
        dados = pd.read_csv(arquivo, sep=";")
        dados = dados[["data", "n_bandas_spearman" if "n_bandas_spearman" in dados else "n_bandas_p_lt_0_05",
                       "r2_loo", "rmse_loo", "n_componentes", "scale", "bandas_nm"]].copy()
        dados.columns = ["data", f"n_bandas_{rotulo}", f"r2_loo_{rotulo}", f"rmse_loo_{rotulo}",
                         f"n_componentes_{rotulo}", f"scale_{rotulo}", f"bandas_nm_{rotulo}"]
        tabelas.append(dados)
    comparacao = tabelas[0].merge(tabelas[1], on="data", validate="one_to_one").merge(
        tabelas[2], on="data", validate="one_to_one"
    )
    comparacao["melhor_limiar_por_r2_loo"] = comparacao[
        ["r2_loo_p_lt_0_001", "r2_loo_p_lt_0_05", "r2_loo_p_lt_0_15"]
    ].idxmax(axis=1).str.replace("r2_loo_", "", regex=False)
    comparacao.to_csv(SAIDA / "comparacao_p001_p005_p015_por_dia.csv", sep=";", index=False)
    print(comparacao[["data", "r2_loo_p_lt_0_001", "r2_loo_p_lt_0_05", "r2_loo_p_lt_0_15",
                      "rmse_loo_p_lt_0_001", "rmse_loo_p_lt_0_05", "rmse_loo_p_lt_0_15",
                      "melhor_limiar_por_r2_loo"]].to_string(index=False))


if __name__ == "__main__":
    main()
