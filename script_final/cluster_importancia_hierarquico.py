"""Agrupa as bandas que entram no Top 5 de importância nos sete dias."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage


PASTA_CLASSIFICACAO = Path(__file__).parent / "outputs" / "classificacao"
DIAS = ("D02", "D03", "D04", "D05", "D06", "D09", "D10")
N_GRUPOS = 4


def carregar_importancias() -> pd.DataFrame:
    """Retorna a matriz banda × dia para a união dos Top 5 diários.

    Zero significa que a banda não esteve entre as cinco mais importantes naquele
    dia; assim, o agrupamento reflete simultaneamente recorrência e intensidade.
    """
    tabela = pd.read_csv(PASTA_CLASSIFICACAO / "top5_rf_por_dia.csv", sep=";")
    matriz = tabela.pivot(index="banda_nm", columns="dia", values="importancia_gini")
    return matriz.reindex(columns=DIAS, fill_value=0).fillna(0).sort_index()


def main() -> None:
    importancia = carregar_importancias()
    valores = importancia.to_numpy()

    # Cada banda é padronizada entre dias: a distância reflete o perfil temporal,
    # não somente a escala absoluta de importância.
    desvios = valores.std(axis=1, ddof=0)
    if (desvios == 0).any():
        raise ValueError("Há bandas com importância constante entre os dias.")
    padronizada = (valores - valores.mean(axis=1, keepdims=True)) / desvios[:, None]

    ligacao = linkage(padronizada, method="ward", metric="euclidean")
    ordem = leaves_list(ligacao)
    grupos = fcluster(ligacao, t=N_GRUPOS, criterion="maxclust")

    resultado = importancia.copy()
    resultado.insert(0, "cluster", grupos)
    resultado.insert(1, "importancia_media", importancia.mean(axis=1))
    resultado.insert(2, "ordem_dendrograma", np.argsort(ordem) + 1)
    resultado.sort_values(["cluster", "importancia_media"], ascending=[True, False]).to_csv(
        PASTA_CLASSIFICACAO / "cluster_hierarquico_top5_importancia_bandas.csv", sep=";"
    )

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    grade = fig.add_gridspec(1, 2, width_ratios=(1.25, 2.75), wspace=0.03)
    ax_dendrograma = fig.add_subplot(grade[0, 0])
    dendrogram(
        ligacao,
        orientation="left",
        labels=importancia.index.astype(str).tolist(),
        leaf_font_size=8,
        color_threshold=0,
        above_threshold_color="#4c4c4c",
        ax=ax_dendrograma,
    )
    ax_dendrograma.set_xlabel("Distância de Ward")
    ax_dendrograma.set_ylabel("Bandas (nm)")

    ax_mapa = fig.add_subplot(grade[0, 1])
    mapa = ax_mapa.imshow(valores[ordem], aspect="auto", cmap="YlOrRd")
    ax_mapa.set_xticks(range(len(DIAS)), DIAS)
    ax_mapa.set_yticks(range(len(ordem)), importancia.index.to_numpy()[ordem])
    ax_mapa.set_yticklabels(importancia.index.to_numpy()[ordem], fontsize=8)
    ax_mapa.yaxis.tick_right()
    ax_mapa.yaxis.set_label_position("right")
    ax_mapa.set_ylabel("Bandas (nm), na ordem do dendrograma")
    ax_mapa.set_title("Importância Gini original")
    fig.colorbar(mapa, ax=ax_mapa, label="Importância Gini", pad=0.02)
    fig.suptitle(
        "Cluster hierárquico das bandas do Top 5 — classificação IRRIG × NIRRIG\n"
        "Perfis de importância do Top 5 padronizados por banda nos sete dias; ligação de Ward"
    )
    fig.savefig(PASTA_CLASSIFICACAO / "cluster_hierarquico_top5_importancia_bandas.png", dpi=220)
    plt.close(fig)

    print(f"Bandas agrupadas: {len(importancia)}")
    print(f"Grupos (corte em {N_GRUPOS}): {pd.Series(grupos).value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
