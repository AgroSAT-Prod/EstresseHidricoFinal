from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 130

OUT = Path("coleta_dados_espectrais/graficos_frequencia")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("dataset/Unificada13052026_Limpa.csv", sep=";", decimal=",")

COLORS = ["#245f73", "#9abf88", "#e0a458", "#c46a5a", "#7a6a9f", "#5a8a9f"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()


def style_counts(ax, counts, title):
    bars = ax.bar(
        counts.index.astype(str),
        counts.to_numpy(),
        color=COLORS[: len(counts)],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{count:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.tick_params(axis="x", rotation=0)


def style_cross(ax, table, col1, col2, title):
    pivot = table.pivot(index=col1, columns=col2, values="amostras").fillna(0)
    pivot.plot(kind="bar", ax=ax, color=COLORS[: pivot.shape[1]], edgecolor="white")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=1)
    ax.set_title(title)
    ax.set_ylim(0, pivot.to_numpy().max() * 1.15)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title=col2, fontsize=9)


ax = axes[0]
style_counts(ax, df["condicao"].value_counts(dropna=False), "1. Condicao")

ax = axes[1]
manha = df[df["turno"].eq("manha")]
counts_dm = manha["data_coleta"].value_counts(dropna=False)


def date_key(v):
    v = str(v)
    return int("".join(c for c in v if c.isdigit()) or 9999)


counts_dm = counts_dm.reindex(sorted(counts_dm.index, key=date_key))
style_counts(ax, counts_dm, "2. Data de coleta (somente manha)")

ax = axes[2]
style_counts(ax, df["genotipo"].value_counts(dropna=False), "3. Somente por genotipo")

ax = axes[3]
table_cg = df.groupby(["condicao", "genotipo"], dropna=False).size().reset_index(name="amostras")
style_cross(ax, table_cg, "condicao", "genotipo", "4. Condicao x genotipo")

ax = axes[4]
table_cgm = (
    manha.groupby(["condicao", "genotipo"], dropna=False)
    .size()
    .reset_index(name="amostras")
)
style_cross(ax, table_cgm, "condicao", "genotipo", "5. Condicao x genotipo (somente manha)")

axes[5].axis("off")

for ax in axes[:5]:
    ax.set_ylabel("Numero de amostras")

fig.suptitle("Frequencia de amostras espectrais", fontsize=15, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out_path = OUT / "frequencias_geral.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Salvo: {out_path}")
