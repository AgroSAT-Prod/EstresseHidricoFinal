from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 110

OUT = Path("coleta_dados_espectrais/graficos_frequencia")
OUT.mkdir(parents=True, exist_ok=True)

dataset_path = Path("dataset/Unificada13052026_Limpa.csv")
df = pd.read_csv(dataset_path, sep=";", decimal=",")

first_band_idx = next(
    idx for idx, col in enumerate(df.columns) if str(col).strip().isdigit()
)
metadata_cols = list(df.columns[:first_band_idx])
band_cols = [col for col in df.columns[first_band_idx:] if str(col).strip().isdigit()]

print("=" * 60)
print("ESTATISTICAS GERAIS")
print("=" * 60)
print(f"Arquivo: {dataset_path}")
print(f"Total de amostras: {len(df):,}".replace(",", "."))
print(f"Total de colunas: {df.shape[1]:,}".replace(",", "."))
print(f"Colunas de metadados ({len(metadata_cols)}): {metadata_cols}")
print(f"Bandas hiperspectrais: {len(band_cols)}")
print(f"Intervalo das bandas: {int(band_cols[0])} a {int(band_cols[-1])} nm")
print(f"Valores ausentes: {int(df.isna().sum().sum()):,}".replace(",", "."))
print(f"Linhas duplicadas: {int(df.duplicated().sum()):,}".replace(",", "."))
print()

print("=" * 60)
print("FREQUENCIA POR METADADO")
print("=" * 60)
counts_by_col = {}
for col in metadata_cols:
    counts = df[col].value_counts(dropna=False)
    if col == "data_coleta":

        def key(v):
            v = str(v)
            return (int("".join(c for c in v if c.isdigit()) or 9999), v)

        counts = counts.reindex(sorted(counts.index, key=key))
    counts_by_col[col] = counts
    print(f"\n--- {col} ({counts.nunique()} valores) ---")
    total = counts.sum()
    for value, count in counts.items():
        print(f"  {value}: {count:,} ({count / total:.1%})".replace(",", "."))
    print(f"  Total: {total:,}".replace(",", "."))

print()
print("=" * 60)
print("CRUZAMENTOS")
print("=" * 60)
crossings = [
    ["condicao", "genotipo"],
    ["condicao", "data_coleta"],
    ["genotipo", "bloco"],
]
for group in crossings:
    avail = [c for c in group if c in df.columns]
    if len(avail) != len(group):
        continue
    print(f"\n--- {' + '.join(group)} ---")
    cross = df.groupby(avail, dropna=False).size().reset_index(name="amostras")
    print(cross.to_string(index=False))

print()
print("=" * 60)
print("GERANDO GRAFICOS DE FREQUENCIA")
print("=" * 60)

COLORS = ["#245f73", "#9abf88", "#e0a458", "#c46a5a", "#7a6a9f", "#5a8a9f"]

for col, counts in counts_by_col.items():
    if len(counts) > 30:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            counts.index.astype(str),
            counts.to_numpy(),
            color=COLORS[0],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(f"Frequencia de amostras por {col} ({len(counts)} categorias)")
        ax.set_xlabel("Numero de amostras")
        ax.set_ylabel(col)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(OUT / f"frequencia_{col}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Salvo: {OUT / f'frequencia_{col}.png'} (horizontal)")
        continue

    fig, ax = plt.subplots(figsize=(9, 4.5))
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
            bar.get_height() + max(counts.max() * 0.01, 1),
            f"{count:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"Frequencia de amostras por {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Numero de amostras")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout()
    fig.savefig(OUT / f"frequencia_{col}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {OUT / f'frequencia_{col}.png'}")

for group in crossings:
    avail = [c for c in group if c in df.columns]
    if len(avail) != len(group):
        continue
    table = df.groupby(avail, dropna=False).size().reset_index(name="amostras")
    pivot = table.pivot(index=avail[0], columns=avail[1], values="amostras").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax, color=COLORS[: pivot.shape[1]], edgecolor="white")
    ax.set_title("Frequencia de amostras por " + " e ".join(avail))
    ax.set_xlabel(avail[0])
    ax.set_ylabel("Numero de amostras")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title=avail[1], fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=1)
    ax.set_ylim(0, pivot.to_numpy().max() * 1.12)
    fig.tight_layout()
    fname = "_".join(avail)
    fig.savefig(OUT / f"frequencia_cruzada_{fname}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {OUT / f'frequencia_cruzada_{fname}.png'}")

print("\nConcluido.")
