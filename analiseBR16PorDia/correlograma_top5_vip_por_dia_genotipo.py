#!/usr/bin/env python3
"""Correlogramas Spearman das Top 5 VIP, por genótipo e dia."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

ENTRADA = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "plsda_vip_5seeds_kfold4"
SAIDA = ROOT / "analiseBR16PorDia" / "dataset_gerado" / "correlograma_top5_vip_por_dia_genotipo"
GENOTIPOS = ("BR16", "CD202", "EMB48")
DIAS = ("D02", "D03", "D04", "D05", "D06", "D09", "D10")


def colunas_bandas(w: np.ndarray, bandas: list[int]) -> list[int]:
    indice = {int(nm): i for i, nm in enumerate(w.astype(int))}
    faltantes = [b for b in bandas if b not in indice]
    if faltantes:
        raise ValueError(f"Bandas não encontradas no espectro: {faltantes}")
    return [indice[b] for b in bandas]


def plotar(matrizes: dict[tuple[str, str], pd.DataFrame]):
    fig, eixos = plt.subplots(len(GENOTIPOS), len(DIAS), figsize=(23, 10.5), constrained_layout=True)
    imagem = None
    for i, genotipo in enumerate(GENOTIPOS):
        for j, dia in enumerate(DIAS):
            ax = eixos[i, j]
            corr = matrizes[genotipo, dia]
            imagem = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
            labels = [str(int(b)) for b in corr.columns]
            ax.set_xticks(range(5), labels, rotation=90, fontsize=7)
            ax.set_yticks(range(5), labels, fontsize=7)
            if i == 0:
                ax.set_title(dia, fontweight="bold")
            if j == 0:
                ax.set_ylabel(genotipo, fontweight="bold", fontsize=10)
            for linha in range(5):
                for coluna in range(5):
                    valor = corr.iat[linha, coluna]
                    ax.text(coluna, linha, f"{valor:.2f}", ha="center", va="center", fontsize=5.8,
                            color="white" if abs(valor) > .55 else "black")
    cbar = fig.colorbar(imagem, ax=eixos, shrink=.78, pad=.012)
    cbar.set_label("Correlação de Spearman (ρ)")
    fig.suptitle("Correlograma das Top 5 bandas VIP por genótipo e dia", fontsize=16, fontweight="bold")
    return fig


def relatorio(resumo: pd.DataFrame) -> str:
    linhas = ["# Correlograma das Top 5 VIP por genótipo e dia", "",
              "Cada matriz usa as cinco bandas selecionadas pela run PLS-DA/VIP do respectivo genótipo e dia. A correlação de Spearman foi calculada entre as leituras da manhã dentro daquela combinação genótipo × dia.",
              "", "A tabela mostra o par de bandas com maior correlação absoluta fora da diagonal em cada matriz.",
              "", "| Genótipo | Dia | Par mais correlacionado (nm) | ρ de Spearman |", "|---|---|---|---:|"]
    for r in resumo.itertuples(index=False):
        linhas.append(f"| {r.genotipo} | {r.dia} | {r.banda_1} × {r.banda_2} | {r.rho_spearman:.3f} |")
    return "\n".join(linhas) + "\n"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    top5 = pd.read_csv(ENTRADA / "top5_bandas_vip_por_dia_genotipo.csv", sep=";")
    meta, espectro, w = carregar("normalizado", turno="manha")
    matrizes, pares, resumo = {}, [], []
    for genotipo in GENOTIPOS:
        for dia in DIAS:
            bandas = (top5.loc[(top5.genotipo.eq(genotipo)) & (top5.dia.eq(dia))]
                      .sort_values("posicao").banda_nm.astype(int).tolist())
            if len(bandas) != 5:
                raise ValueError(f"Top 5 incompleto em {genotipo}/{dia}: {bandas}")
            mascara = (meta.genotipo.eq(genotipo) & meta.dia.eq(dia)).to_numpy()
            dados = espectro[mascara][:, colunas_bandas(w, bandas)]
            corr = pd.DataFrame(dados, columns=bandas).corr(method="spearman")
            matrizes[genotipo, dia] = corr
            corr.to_csv(SAIDA / f"correlacao_spearman_{genotipo}_{dia}.csv", sep=";", index_label="banda_nm")
            for a in range(len(bandas)):
                for b in range(a + 1, len(bandas)):
                    pares.append({"genotipo": genotipo, "dia": dia, "banda_1": bandas[a], "banda_2": bandas[b], "rho_spearman": corr.iat[a, b], "rho_abs": abs(corr.iat[a, b])})
            maior = max(pares[-10:], key=lambda x: x["rho_abs"])
            resumo.append({k: maior[k] for k in ("genotipo", "dia", "banda_1", "banda_2", "rho_spearman", "rho_abs")})
    df_pares, df_resumo = pd.DataFrame(pares), pd.DataFrame(resumo)
    df_pares.to_csv(SAIDA / "correlacoes_pares_top5.csv", sep=";", index=False)
    df_resumo.to_csv(SAIDA / "resumo_maior_correlacao_por_genotipo_dia.csv", sep=";", index=False)
    (SAIDA / "relatorio_correlograma.md").write_text(relatorio(df_resumo), encoding="utf-8")
    pd.DataFrame([{"fonte_bandas": str(ENTRADA / "top5_bandas_vip_por_dia_genotipo.csv"), "metodo": "Spearman", "turno": "manha", "genotipos": ", ".join(GENOTIPOS), "dias": ", ".join(DIAS), "bandas_por_matriz": 5}]).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)
    fig = plotar(matrizes)
    fig.savefig(SAIDA / "correlograma_top5_vip_por_dia_genotipo.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "correlograma_top5_vip_por_dia_genotipo.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"21 matrizes salvas em {SAIDA}")


if __name__ == "__main__":
    main()
