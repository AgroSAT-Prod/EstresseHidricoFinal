"""Correlação de Spearman das bandas Top 5 da classificação nos sete dias."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    pasta = Path(__file__).resolve().parent
    notebook = json.loads((pasta / "Pipeline_Final_Classificacao.ipynb").read_text())
    # Executa as células de pré-processamento do próprio pipeline, sem duplicar regras.
    contexto = {"__name__": "__correlacao__"}
    for numero in (1, 3, 4, 6, 7):
        exec(compile("".join(notebook["cells"][numero]["source"]),
                     f"Pipeline_Final_Classificacao.ipynb:célula{numero}", "exec"), contexto)

    saida = pasta / "outputs" / "classificacao"
    top = pd.read_csv(saida / "top5_rf_por_dia.csv", sep=";")
    dias = sorted(top["dia"].unique())
    bandas = sorted(top["banda_nm"].astype(int).unique())
    meta = contexto["base_limpa"]
    dia = meta["data_coleta"].astype(str).str.extract(r"^(D\d+)")[0]
    mascara = meta["turno"].eq("manha") & dia.isin(dias)
    dados = contexto["base_normalizada"].loc[mascara, [str(b) for b in bandas]]
    if dados.empty or dados.isna().any().any():
        raise ValueError("Dados ausentes ou nenhuma amostra selecionada.")
    correlacao = dados.corr(method="spearman")
    correlacao.to_csv(saida / "correlacao_spearman_top5_classificacao.csv", sep=";", float_format="%.8f")

    fig, ax = plt.subplots(figsize=(15, 13), constrained_layout=True)
    mapa = ax.imshow(correlacao, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(bandas)), bandas, rotation=90)
    ax.set_yticks(np.arange(len(bandas)), bandas)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Comprimento de onda (nm)")
    ax.set_title(f"Correlação de Spearman entre bandas do Top 5 da classificação\n"
                 f"Sete dias reunidos • leituras da manhã • n = {len(dados)}")
    fig.colorbar(mapa, ax=ax, label="Correlação de Spearman (ρ)", shrink=0.85)
    fig.savefig(saida / "correlacao_spearman_top5_classificacao.png", dpi=220)
    plt.close(fig)
    print(f"Amostras: {len(dados)}; bandas: {len(bandas)}; dias: {', '.join(dias)}")
    print(dia[mascara].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
