#!/usr/bin/env python3
"""Painel dos testes de diferenca significativa (Kruskal-Wallis + Dunn).

Reune o que os dois scripts do modulo produzem, dia a dia:

A) Bandas significativas por dia para cada fator -- genotipo, condicao e o
   omnibus das 6 celulas. Expoe a inversao de D06 e D09, em que a condicao
   separa quase todo o espectro enquanto o genotipo quase nao separa nada.
B) Efeito do estresse dentro de cada genotipo (IRRIG vs NIRRIG), em duas
   leituras: quantas bandas separam, e quao forte e a separacao (delta de
   Cliff). Com o n deste experimento quase tudo da significativo, entao e a
   segunda leitura que ordena os resultados.
C) Os 15 pares de celulas ao longo dos sete dias, agrupados pelo tipo de
   pergunta que respondem: efeito do estresse (3 pares), separacao entre
   genotipos na mesma condicao (6) e pares cruzados (6).
D) Matriz 6x6 de separacao entre as celulas, uma por dia.
E) Onde no espectro o contraste de estresse se concentra, por regiao.

Todas as analises usam apenas o turno da manha.

Paleta validada por codigo contra deuteranopia, protanopia e tritanopia.

Uso:
    python plot_diferencas.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
SAIDA_DIR = ROOT / "dataset_gerado"
SAIDA = ROOT / "diferencas_painel.png"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]

# Mesma paleta validada da figura-resumo: 10 pares, pior dE 21.6 em visao
# normal e 10.0 sob CVD (pisos 15 e 8).
COR_GENOTIPO = {"BR16": "#1baf7a", "CD202": "#ad1457", "EMB48": "#1a237e"}
COR_CONDICAO = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}

COR_FATOR = {
    "sig_condicao": "#eb6834",
    "sig_genotipo": "#1baf7a",
    "sig_omnibus": "#52514e",
}
ROTULO_FATOR = {
    "sig_condicao": "Condicao (IRRIG vs NIRRIG)",
    "sig_genotipo": "Genotipo (3 grupos)",
    "sig_omnibus": "Omnibus (6 celulas)",
}

COR_TEXTO = "#1a1a19"
COR_TEXTO_FRACO = "#52514e"

CMAP_SEQUENCIAL = LinearSegmentedColormap.from_list(
    "magnitude", ["#f4f6fa", "#7fa8d9", "#1a3a6b"]
)

# Ordem fixa dos tipos de par, do mais direto ao mais confundido.
TIPOS = ["estresse", "genotipo", "cruzado"]
ROTULO_TIPO = {"estresse": "ESTRESSE", "genotipo": "GENOTIPO", "cruzado": "CRUZADO"}

REGIOES = ["VIS", "red edge", "NIR", "SWIR1", "SWIR2"]


def painel_a(ax: plt.Axes, resumo: pd.DataFrame) -> None:
    """Bandas significativas por dia, para cada fator."""
    dias = list(resumo["dia"])
    x = np.arange(len(dias))
    total = int(resumo["bandas"].iloc[0])

    for coluna, cor in COR_FATOR.items():
        ax.plot(x, resumo[coluna], color=cor, linewidth=2, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                label=ROTULO_FATOR[coluna])

    ax.axhline(total, color=COR_TEXTO_FRACO, linestyle=":", linewidth=1,
               label=f"Todas as {total} bandas")

    # A inversao que o painel existe para mostrar: em D06 e D09 a condicao
    # separa quase todo o espectro enquanto o genotipo quase nao separa nada.
    indices = [dias.index(d) for d in ("D06", "D09") if d in dias]
    if indices:
        meio = sum(indices) / len(indices)
        alvo = min(resumo["sig_genotipo"].iloc[i] for i in indices)
        ax.annotate("condicao domina,\ngenotipo some",
                    xy=(meio, alvo), xytext=(meio, total * 0.42),
                    ha="center", va="center", fontsize=8,
                    color=COR_TEXTO_FRACO,
                    arrowprops=dict(arrowstyle="->", color=COR_TEXTO_FRACO,
                                    linewidth=0.8,
                                    connectionstyle="arc3,rad=0.2"))

    ax.set_xticks(x)
    ax.set_xticklabels(dias, fontsize=8)
    ax.set_ylim(-40, total * 1.08)
    ax.set_xlabel("Dia de coleta", fontsize=9)
    ax.set_ylabel("Bandas significativas (q < 0.05)", fontsize=9)
    ax.set_title("A) Kruskal-Wallis por dia: qual fator separa o espectro",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.3)


def painel_b(axes: list[plt.Axes], estresse: pd.DataFrame) -> None:
    """Efeito do estresse por genotipo: extensao e magnitude."""
    dias = sorted(estresse["dia"].unique())
    x = np.arange(len(dias))

    medidas = [
        ("significativa", "Bandas com IRRIG != NIRRIG (%)",
         "B1) Extensao do efeito", 100.0),
        ("delta_cliff", "|delta de Cliff| mediano",
         "B2) Magnitude do efeito", 1.0),
    ]

    for ax, (coluna, rotulo_y, titulo, escala) in zip(axes, medidas):
        for genotipo in GENOTIPOS:
            sub = estresse[estresse["genotipo"] == genotipo]
            if coluna == "significativa":
                y = sub.groupby("dia")[coluna].mean().reindex(dias).to_numpy() * escala
            else:
                y = (
                    sub.assign(abs_delta=sub[coluna].abs())
                    .groupby("dia")["abs_delta"].median().reindex(dias).to_numpy()
                )
            ax.plot(x, y, color=COR_GENOTIPO[genotipo], linewidth=2, marker="o",
                    markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                    label=genotipo)
            ax.annotate(genotipo, xy=(x[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        fontweight="bold", color=COR_GENOTIPO[genotipo])

        ax.set_xticks(x)
        ax.set_xticklabels(dias, fontsize=8)
        ax.set_xlim(-0.3, len(dias) - 0.15)
        ax.set_xlabel("Dia de coleta", fontsize=9)
        ax.set_ylabel(rotulo_y, fontsize=9)
        ax.set_title(titulo, fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(-3, 108)
    axes[0].legend(fontsize=7.5, loc="lower right", title="Genotipo",
                   title_fontsize=7.5)


def painel_c(ax: plt.Axes, cax: plt.Axes, resumo: pd.DataFrame) -> None:
    """Os 15 pares ao longo dos dias, agrupados por tipo de pergunta."""
    dias = sorted(resumo["dia"].unique())

    ordem = (
        resumo[["celula_a", "celula_b", "tipo"]].drop_duplicates()
        .assign(ordem=lambda d: d["tipo"].map({t: i for i, t in enumerate(TIPOS)}))
        .sort_values(["ordem", "celula_a", "celula_b"])
    )
    rotulos = [f"{a}  vs  {b}" for a, b in zip(ordem["celula_a"], ordem["celula_b"])]

    grade = np.full((len(ordem), len(dias)), np.nan)
    for i, (a, b) in enumerate(zip(ordem["celula_a"], ordem["celula_b"])):
        sub = resumo[(resumo["celula_a"] == a) & (resumo["celula_b"] == b)]
        serie = sub.set_index("dia")["prop_sig"].reindex(dias) * 100
        grade[i] = serie.to_numpy()

    im = ax.imshow(grade, cmap=CMAP_SEQUENCIAL, vmin=0, vmax=100, aspect="auto")

    for i in range(len(ordem)):
        for j in range(len(dias)):
            if np.isnan(grade[i, j]):
                continue
            ax.text(j, i, f"{grade[i, j]:.0f}", ha="center", va="center",
                    fontsize=7.5, color="white" if grade[i, j] > 55 else COR_TEXTO)

    # Separadores entre os blocos de tipo.
    limites = np.cumsum(ordem["tipo"].value_counts().reindex(TIPOS).to_numpy())
    for limite in limites[:-1]:
        ax.axhline(limite - 0.5, color="white", linewidth=3)

    inicio = 0
    for tipo, fim in zip(TIPOS, limites):
        ax.annotate(ROTULO_TIPO[tipo], xy=(-0.245, (inicio + fim - 1) / 2),
                    xycoords=("axes fraction", "data"), ha="center", va="center",
                    rotation=90, fontsize=9, fontweight="bold",
                    color=COR_TEXTO_FRACO)
        inicio = fim

    ax.set_xticks(range(len(dias)))
    ax.set_xticklabels(dias, fontsize=8)
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels(rotulos, fontsize=7.5)
    ax.set_title("C) Os 15 pares de celulas ao longo dos dias "
                 "(% de bandas separadas, pos-hoc de Dunn)\n"
                 "     ESTRESSE: X IRRIG vs X NIRRIG    "
                 "GENOTIPO: materiais diferentes na mesma condicao    "
                 "CRUZADO: genotipo e condicao variando juntos",
                 fontsize=10, fontweight="bold", loc="left")
    ax.grid(False)

    barra = plt.colorbar(im, cax=cax)
    barra.set_label("Bandas separadas (%)", fontsize=8)
    barra.ax.tick_params(labelsize=7)


def painel_d(axes: list[plt.Axes], resumo: pd.DataFrame) -> None:
    """Matriz 6x6 de separacao entre as celulas, uma por dia."""
    celulas = [f"{g}|{c}" for g in GENOTIPOS for c in CONDICOES]
    curto = [f"{g[:5]}\n{c[0]}" for g in GENOTIPOS for c in CONDICOES]
    indice = {nome: i for i, nome in enumerate(celulas)}
    dias = sorted(resumo["dia"].unique())

    for ax, dia in zip(axes, dias):
        grade = np.full((len(celulas), len(celulas)), np.nan)
        for _, row in resumo[resumo["dia"] == dia].iterrows():
            i, j = indice[row["celula_a"]], indice[row["celula_b"]]
            grade[i, j] = grade[j, i] = row["prop_sig"] * 100

        ax.imshow(grade, cmap=CMAP_SEQUENCIAL, vmin=0, vmax=100)
        for i in range(len(celulas)):
            for j in range(len(celulas)):
                if np.isnan(grade[i, j]):
                    continue
                ax.text(j, i, f"{grade[i, j]:.0f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if grade[i, j] > 55 else COR_TEXTO)

        ax.set_xticks(range(len(celulas)))
        ax.set_yticks(range(len(celulas)))
        ax.set_xticklabels(curto, fontsize=5)
        ax.set_yticklabels(curto if ax is axes[0] else [], fontsize=5)
        for i, nome in enumerate(celulas):
            cor = COR_GENOTIPO[nome.split("|")[0]]
            ax.get_xticklabels()[i].set_color(cor)
            if ax is axes[0]:
                ax.get_yticklabels()[i].set_color(cor)
        ax.set_title(dia, fontsize=9, fontweight="bold")
        ax.grid(False)

    axes[0].text(0.0, 1.30, "D) Separacao entre as 6 celulas, dia a dia "
                 "(I = IRRIG, N = NIRRIG)",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold",
                 ha="left", va="bottom")


def painel_e(axes: list[plt.Axes], regioes: pd.DataFrame, cax: plt.Axes) -> None:
    """Onde no espectro o contraste de estresse se concentra."""
    estresse = regioes[regioes["tipo"] == "estresse"].copy()
    estresse["genotipo"] = estresse["celula_a"].str.split("|").str[0]
    dias = sorted(estresse["dia"].unique())

    for ax, genotipo in zip(axes, GENOTIPOS):
        sub = estresse[estresse["genotipo"] == genotipo]
        grade = (
            sub.pivot(index="regiao", columns="dia", values="prop_sig")
            .reindex(index=REGIOES, columns=dias) * 100
        )
        im = ax.imshow(grade.to_numpy(), cmap=CMAP_SEQUENCIAL, vmin=0, vmax=100,
                       aspect="auto")

        for i in range(len(REGIOES)):
            for j in range(len(dias)):
                valor = grade.to_numpy()[i, j]
                if np.isnan(valor):
                    continue
                ax.text(j, i, f"{valor:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if valor > 55 else COR_TEXTO)

        ax.set_xticks(range(len(dias)))
        ax.set_xticklabels(dias, fontsize=7.5)
        ax.set_yticks(range(len(REGIOES)))
        ax.set_yticklabels(REGIOES if ax is axes[0] else [], fontsize=7.5)
        ax.set_title(genotipo, fontsize=9, fontweight="bold",
                     color=COR_GENOTIPO[genotipo])
        ax.grid(False)

    axes[0].text(0.0, 1.22, "E) Em que regiao do espectro o estresse separa "
                 "IRRIG de NIRRIG (% de bandas da regiao)",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold",
                 ha="left", va="bottom")

    barra = plt.colorbar(im, cax=cax)
    barra.set_label("Bandas separadas (%)", fontsize=8)
    barra.ax.tick_params(labelsize=7)


def gerar(
    resumo_kw: pd.DataFrame,
    estresse: pd.DataFrame,
    resumo_pares: pd.DataFrame,
    regioes: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(18, 21))
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[1.0, 1.6, 0.62, 0.72],
        hspace=0.42, wspace=1.5,
        top=0.94, bottom=0.035, left=0.055, right=0.95,
    )

    ax_a = fig.add_subplot(gs[0, :5])
    painel_a(ax_a, resumo_kw)

    gs_b = gs[0, 6:].subgridspec(1, 2, wspace=0.42)
    axes_b = [fig.add_subplot(gs_b[i]) for i in range(2)]
    painel_b(axes_b, estresse)

    ax_c = fig.add_subplot(gs[1, 3:11])
    cax_c = fig.add_subplot(gs[1, 11])
    painel_c(ax_c, cax_c, resumo_pares)

    gs_d = gs[2, :].subgridspec(1, 7, wspace=0.18)
    axes_d = [fig.add_subplot(gs_d[i]) for i in range(7)]
    painel_d(axes_d, resumo_pares)

    gs_e = gs[3, :11].subgridspec(1, 3, wspace=0.14)
    axes_e = [fig.add_subplot(gs_e[i]) for i in range(3)]
    cax_e = fig.add_subplot(gs[3, 11])
    painel_e(axes_e, regioes, cax_e)

    fig.suptitle(
        "Testes de diferenca significativa: Kruskal-Wallis e pos-hoc de Dunn, "
        "dia a dia\n"
        "3 genotipos x 2 condicoes, turno da manha -- 192 leituras por dia, "
        "2051 bandas, FDR de Benjamini-Hochberg",
        fontsize=15, fontweight="bold", y=0.982,
    )

    fig.savefig(SAIDA, dpi=300)
    print(f"Painel salvo em: {SAIDA}")
    print("  Resolucao: 300 DPI")
    print("  Tamanho: 18x23 pol")


def main() -> None:
    print("Gerando painel dos testes de diferenca significativa...\n")

    resumo_kw = pd.read_csv(SAIDA_DIR / "diferencas_resumo.csv", sep=";")
    estresse = pd.read_csv(SAIDA_DIR / "comparacao_estresse.csv", sep=";")
    resumo_pares = pd.read_csv(SAIDA_DIR / "comparacao_resumo.csv", sep=";")
    regioes = pd.read_csv(SAIDA_DIR / "comparacao_regioes.csv", sep=";")

    print(f"  diferencas_resumo: {len(resumo_kw)} dias")
    print(f"  comparacao_estresse: {len(estresse)} linhas")
    print(f"  comparacao_resumo: {len(resumo_pares)} pares x dia")
    print(f"  comparacao_regioes: {len(regioes)} linhas\n")

    gerar(resumo_kw, estresse, resumo_pares, regioes)
    print("Concluido.")


if __name__ == "__main__":
    main()
