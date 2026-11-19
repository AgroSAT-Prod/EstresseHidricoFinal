"""Generate the English version of the temporal Random Forest band plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PASTA_CLASSIFICACAO = Path(__file__).resolve().parent / "outputs" / "classificacao"
top5_rf_por_dia = pd.read_csv(PASTA_CLASSIFICACAO / "top5_rf_por_dia.csv", sep=";")

# One row per band, grouped by day and ranked by Gini importance.
top5_grafico = top5_rf_por_dia.sort_values(["dia", "posicao"]).copy()
dias_importancia = sorted(top5_grafico["dia"].unique())
posicao_dia = {dia: indice for indice, dia in enumerate(dias_importancia)}
top5_grafico["linha"] = top5_grafico["dia"].map(posicao_dia) * 6 + top5_grafico["posicao"]

fig, eixo = plt.subplots(
    figsize=(13, max(5, 1.65 * len(dias_importancia) + 1.5)),
    constrained_layout=True,
)
pontos = eixo.scatter(
    top5_grafico["banda_nm"], top5_grafico["linha"],
    c=top5_grafico["importancia_gini"], cmap="viridis", s=105,
    edgecolors="#333333", linewidths=0.5, zorder=3,
)
for linha in top5_grafico.itertuples():
    eixo.annotate(
        f"{int(linha.banda_nm)} nm",
        (linha.banda_nm, linha.linha),
        xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=8,
    )
for indice in range(1, len(dias_importancia)):
    eixo.axhline(indice * 6 + 0.5, color="#dddddd", linewidth=0.8)
eixo.set_yticks([i * 6 + 3 for i in range(len(dias_importancia))], labels=dias_importancia)
eixo.set_ylim(len(dias_importancia) * 6 + 0.5, 0)
eixo.set_xlim(380, 2470)
eixo.set_xlabel("Wavelength (nm)")
eixo.set_ylabel("Day (bands ranked by importance)")

eixo.grid(axis="x", alpha=0.25)
fig.colorbar(pontos, ax=eixo, label="Gini importance")
arquivo_importancias = PASTA_CLASSIFICACAO / "importancia_top5_por_dia_en.png"
fig.savefig(arquivo_importancias, dpi=180, bbox_inches="tight")
plt.close(fig)
print(arquivo_importancias)
