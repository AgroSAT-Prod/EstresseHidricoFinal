"""Compara o FWHM das curvas médias por dia e condição."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PASTA = Path(__file__).resolve().parent / "outputs/classificacao"
dados = pd.read_csv(PASTA / "fwhm_regioes_por_dia.csv", sep=";", decimal=",")
fig, eixos = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
for regiao, eixo in zip(sorted(dados["regiao"].unique()), eixos.flat):
    quadro = dados.loc[dados["regiao"].eq(regiao)]
    for condicao, nome, cor in [("IRRIG", "Irrigado", "#2166ac"), ("NIRRIG", "Não irrigado", "#d95f02")]:
        grupo = quadro.loc[quadro["condicao"].eq(condicao)].sort_values("dia")
        eixo.plot(grupo["dia"], grupo["fwhm_nm"], marker="o", linewidth=2, label=nome, color=cor)
    linha = quadro.iloc[0]
    eixo.set_title(f"Região {regiao}: {int(linha['extremo_esquerdo_nm'])}–{int(linha['extremo_direito_nm'])} nm")
    eixo.set_ylabel("FWHM (nm)")
    eixo.set_ylim(bottom=0)
    eixo.grid(alpha=0.25)
    eixo.legend()
fig.supxlabel("Dia de coleta")
fig.suptitle("Largura à meia altura das curvas espectrais médias", fontsize=15)
fig.savefig(PASTA / "fwhm_regioes_por_dia.png", dpi=200)
