#!/usr/bin/env python3
"""FDR e VIP das bandas Top 5 diarias, reunidas por genotipo.

Para cada genotipo, parte das 35 selecoes (Top 5 de sete dias). Bandas que se
repetem entre dias sao testadas uma unica vez. O teste de Mann--Whitney entre
IRRIG e NIRRIG e corrigido por Benjamini--Hochberg dentro de cada genotipo.
Somente bandas com q-FDR <= 0,05 entram no PLS-DA que gera o ranking VIP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cross_decomposition import PLSRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

ENTRADA = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "plsda_vip_5seeds_kfold4"
SAIDA = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "fdr_vip_todos_dias_por_genotipo"
GENOTIPOS = ("BR16", "CD202", "EMB48")
DIAS = ("D02", "D03", "D04", "D05", "D06", "D09", "D10")
ALFA, TOP_K, N_COMPONENTES = 0.05, 5, 2
CORES = {"BR16": "#4C78A8", "CD202": "#F58518", "EMB48": "#54A24B"}


def calcular_vip(modelo: PLSRegression) -> np.ndarray:
    t, w, q = modelo.x_scores_, modelo.x_weights_, modelo.y_loadings_
    soma = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
    total = soma.sum()
    return np.sqrt(w.shape[0] * ((w ** 2) @ soma) / total) if total else np.zeros(w.shape[0])


def benjamini_hochberg(p_valores: np.ndarray) -> np.ndarray:
    """Valores q ajustados pelo procedimento FDR de Benjamini--Hochberg."""
    p = np.asarray(p_valores, dtype=float)
    ordem = np.argsort(p)
    ordenados = p[ordem]
    n = len(p)
    ajustados = np.minimum.accumulate((ordenados * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[ordem] = np.minimum(ajustados, 1.0)
    return q


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    meta, X, comprimentos = carregar("normalizado", turno="manha")
    origem = pd.read_csv(ENTRADA / "top5_bandas_vip_por_dia_genotipo.csv", sep=";")
    indice_nm = {int(nm): i for i, nm in enumerate(comprimentos.astype(int))}
    todos_testes, todos_vips, todos_top, configuracoes = [], [], [], []

    for genotipo in GENOTIPOS:
        selecoes = origem.loc[origem.genotipo.eq(genotipo) & origem.dia.isin(DIAS)]
        bandas = np.sort(selecoes.banda_nm.astype(int).unique())
        mascara = (meta.genotipo.eq(genotipo) & meta.dia.isin(DIAS)).to_numpy()
        metag, Xg = meta.loc[mascara].reset_index(drop=True), X[mascara]
        y = metag.condicao.eq("NIRRIG").to_numpy()
        idx = np.array([indice_nm[b] for b in bandas])

        testes = []
        for banda, coluna in zip(bandas, Xg[:, idx].T):
            resultado = mannwhitneyu(coluna[y], coluna[~y], alternative="two-sided", method="asymptotic")
            testes.append({"banda_nm": banda, "estatistica_u": resultado.statistic, "p_valor": resultado.pvalue})
        testes = pd.DataFrame(testes)
        testes["q_fdr"] = benjamini_hochberg(testes.p_valor.to_numpy())
        testes["significativa_fdr"] = testes.q_fdr <= ALFA
        testes.insert(0, "genotipo", genotipo)
        testes = testes.sort_values(["q_fdr", "p_valor", "banda_nm"]).reset_index(drop=True)
        testes.to_csv(SAIDA / f"fdr_bandas_{genotipo}.csv", sep=";", index=False)
        todos_testes.append(testes)

        selecionadas = testes.loc[testes.significativa_fdr, "banda_nm"].to_numpy(dtype=int)
        if not len(selecionadas):
            raise RuntimeError(f"Nenhuma banda passou FDR para {genotipo}.")
        Xs = Xg[:, [indice_nm[b] for b in selecionadas]]
        modelo = PLSRegression(n_components=min(N_COMPONENTES, Xs.shape[1], Xs.shape[0] - 1), scale=True).fit(Xs, y.astype(float))
        vips = pd.DataFrame({"banda_nm": selecionadas, "vip": calcular_vip(modelo)})
        vips = vips.merge(testes[["banda_nm", "p_valor", "q_fdr"]], on="banda_nm", how="left")
        vips = vips.sort_values("vip", ascending=False).reset_index(drop=True)
        vips.insert(0, "posicao", np.arange(1, len(vips) + 1))
        vips.insert(0, "genotipo", genotipo)
        vips.to_csv(SAIDA / f"vip_bandas_fdr_{genotipo}.csv", sep=";", index=False)
        todos_vips.append(vips)
        todos_top.append(vips.head(TOP_K))
        configuracoes.append({"genotipo": genotipo, "selecoes_top5_diarias": len(selecoes), "bandas_unicas_testadas": len(bandas), "bandas_q_fdr_le_0_05": len(selecionadas), "amostras": len(metag), "teste": "Mann-Whitney bilateral", "correcao": "Benjamini-Hochberg", "alfa_fdr": ALFA, "modelo": "PLS-DA", "componentes": modelo.n_components})

    top = pd.concat(todos_top, ignore_index=True)
    pd.concat(todos_testes, ignore_index=True).to_csv(SAIDA / "fdr_bandas_por_genotipo.csv", sep=";", index=False)
    pd.concat(todos_vips, ignore_index=True).to_csv(SAIDA / "vip_bandas_fdr_por_genotipo.csv", sep=";", index=False)
    top.to_csv(SAIDA / "top5_vip_apos_fdr_por_genotipo.csv", sep=";", index=False)
    config = pd.DataFrame(configuracoes)
    config.to_csv(SAIDA / "configuracao.csv", sep=";", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
    for ax, genotipo in zip(axes, GENOTIPOS):
        dados = top.loc[top.genotipo.eq(genotipo)].sort_values("vip")
        ax.barh(dados.banda_nm.astype(str), dados.vip, color=CORES[genotipo])
        ax.axvline(1, color="#555", linestyle="--", linewidth=1)
        ax.set_title(genotipo, fontweight="bold", color=CORES[genotipo])
        ax.set_xlabel("VIP")
        ax.grid(axis="x", linestyle=":", alpha=.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Comprimento de onda (nm)")
    fig.suptitle("Top 5 VIP apos FDR (Benjamini-Hochberg), por genotipo", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(SAIDA / "top5_vip_apos_fdr_por_genotipo.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "top5_vip_apos_fdr_por_genotipo.pdf", bbox_inches="tight")
    plt.close(fig)

    linhas = ["# FDR seguido de PLS-DA/VIP por genotipo", "", "As candidatas foram as 35 selecoes Top 5 das sete analises diarias por genotipo. Bandas repetidas foram deduplicadas antes do teste. Para cada genotipo, Mann-Whitney bilateral (IRRIG x NIRRIG; sete dias, manha) foi corrigido por Benjamini-Hochberg; apenas q-FDR <= 0,05 entrou no PLS-DA/VIP.", "", "| Genotipo | Bandas unicas testadas | Passaram FDR | Top 5 (nm; VIP; q-FDR) |", "|---|---:|---:|---|"]
    for g in GENOTIPOS:
        c = config.loc[config.genotipo.eq(g)].iloc[0]
        valores = top.loc[top.genotipo.eq(g)]
        ranking = ", ".join(f"{int(x.banda_nm)} ({x.vip:.3f}; {x.q_fdr:.2e})" for x in valores.itertuples())
        linhas.append(f"| {g} | {c.bandas_unicas_testadas} | {c.bandas_q_fdr_le_0_05} | {ranking} |")
    (SAIDA / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
