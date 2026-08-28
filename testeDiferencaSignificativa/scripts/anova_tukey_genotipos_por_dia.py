#!/usr/bin/env python3
"""ANOVA de uma via + Tukey HSD entre genótipos, por dia e condição.

Em cada combinação de dia e condição, testa BR16, CD202 e EMB48 em cada
banda do espectro normalizado. Bandas com ANOVA p <= 0,05 seguem para o
pós-teste de Tukey HSD; um par é considerado diferente quando p de Tukey
(já ajustado para os três pares) <= 0,05.

Saídas em ``dataset_gerado``:
* anova_genotipos_por_dia.csv: resultado omnibus para todas as bandas;
* tukey_genotipos_por_dia.csv: três pares por banda com ANOVA significativa;
* anova_tukey_genotipos_resumo.csv: contagens por dia e condição.
* anova_tukey_genotipos_resumo_pares.csv: contagens por par de genótipos.

Uso:
    python anova_tukey_genotipos_por_dia.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import studentized_range

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402
from diferencas_significativas import CONDICOES, GENOTIPOS  # noqa: E402

SAIDA_DIR = ROOT.parent / "resultados" / "dataset_gerado"
ESTAGIO_PADRAO = "normalizado"
TURNO = "manha"
ALPHA = 0.05


def anova_uma_via(grupos: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F, p e quadrado eta da ANOVA de uma via, banda a banda."""
    n_grupos = len(grupos)
    tamanhos = np.array([len(g) for g in grupos])
    n = tamanhos.sum()
    medias = np.array([g.mean(axis=0) for g in grupos])
    media_geral = np.average(medias, axis=0, weights=tamanhos)
    ss_entre = ((medias - media_geral) ** 2 * tamanhos[:, None]).sum(axis=0)
    ss_dentro = sum(((g - media) ** 2).sum(axis=0) for g, media in zip(grupos, medias))
    gl_entre, gl_dentro = n_grupos - 1, n - n_grupos
    with np.errstate(divide="ignore", invalid="ignore"):
        f_valor = (ss_entre / gl_entre) / (ss_dentro / gl_dentro)
        eta2 = ss_entre / (ss_entre + ss_dentro)
    p_valor = f_dist.sf(f_valor, gl_entre, gl_dentro)
    return f_valor, p_valor, eta2


def tukey_hsd(
    grupos: list[np.ndarray],
    nomes: list[str],
    bandas: np.ndarray,
    f_valor: np.ndarray,
    p_anova: np.ndarray,
) -> pd.DataFrame:
    """Tukey HSD para as bandas cujo teste omnibus foi significativo.

    O limiar crítico do alcance studentizado é obtido para alpha=0,05; assim,
    ``q_tukey >= q_critico_tukey`` é exatamente equivalente a p_Tukey <= 0,05.
    Evita-se calcular um p-valor individual por banda, operação muito lenta
    para as 2.051 bandas e desnecessária para o critério solicitado.
    """
    selecionadas = np.flatnonzero(p_anova <= ALPHA)
    if not len(selecionadas):
        return pd.DataFrame(columns=[
            "banda_nm", "genotipo_a", "genotipo_b", "diferenca_media",
            "q_tukey", "q_critico_tukey", "significativo_tukey", "F_anova", "p_anova",
        ])

    tamanhos = np.array([len(g) for g in grupos])
    n_total, k = tamanhos.sum(), len(grupos)
    gl_dentro = n_total - k
    medias = np.array([g.mean(axis=0) for g in grupos])
    ss_dentro = sum(((g - media) ** 2).sum(axis=0) for g, media in zip(grupos, medias))
    qm_dentro = ss_dentro / gl_dentro

    q_critico = studentized_range.isf(ALPHA, k, gl_dentro)
    linhas = []
    for i, j in combinations(range(k), 2):
        diferenca = medias[i, selecionadas] - medias[j, selecionadas]
        erro = np.sqrt(qm_dentro[selecionadas] / 2 * (1 / tamanhos[i] + 1 / tamanhos[j]))
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.abs(diferenca) / erro
        linhas.append(pd.DataFrame({
            "banda_nm": bandas[selecionadas].astype(int),
            "genotipo_a": nomes[i],
            "genotipo_b": nomes[j],
            "diferenca_media": diferenca,
            "q_tukey": q,
            "q_critico_tukey": q_critico,
            "significativo_tukey": q >= q_critico,
            "F_anova": f_valor[selecionadas],
            "p_anova": p_anova[selecionadas],
        }))
    return pd.concat(linhas, ignore_index=True)


def analisar_celula(Y: np.ndarray, meta: pd.DataFrame, bandas: np.ndarray, dia: str, condicao: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = meta[(meta["dia"] == dia) & (meta["condicao"] == condicao)]
    idx = sub.index.to_numpy()
    grupos = [Y[idx[sub["genotipo"].to_numpy() == genotipo]] for genotipo in GENOTIPOS]
    if any(len(g) < 2 for g in grupos):
        raise ValueError(f"Amostras insuficientes em {dia} × {condicao}.")

    f_valor, p_anova, eta2 = anova_uma_via(grupos)
    anova = pd.DataFrame({
        "dia": dia, "condicao": condicao, "banda_nm": bandas.astype(int),
        "n_BR16": len(grupos[0]), "n_CD202": len(grupos[1]), "n_EMB48": len(grupos[2]),
        "F_anova": f_valor, "p_anova": p_anova, "eta2": eta2,
        "significativo_anova": p_anova <= ALPHA,
    })
    tukey = tukey_hsd(grupos, GENOTIPOS, bandas, f_valor, p_anova)
    tukey.insert(0, "condicao", condicao)
    tukey.insert(0, "dia", dia)
    return anova, tukey


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    meta, Y, bandas = carregar(estagio, turno=TURNO)

    anovas, tukeys = [], []
    for dia in sorted(meta["dia"].unique()):
        for condicao in CONDICOES:
            anova, tukey = analisar_celula(Y, meta, bandas, dia, condicao)
            anovas.append(anova)
            tukeys.append(tukey)

    df_anova = pd.concat(anovas, ignore_index=True)
    df_tukey = pd.concat(tukeys, ignore_index=True)
    resumo = (
        df_anova.groupby(["dia", "condicao"], as_index=False)
        .agg(bandas_testadas=("banda_nm", "size"),
             bandas_sig_anova=("significativo_anova", "sum"))
    )
    pares = (
        df_tukey[df_tukey["significativo_tukey"]]
        .groupby(["dia", "condicao"], as_index=False)
        .agg(pares_sig_tukey=("significativo_tukey", "sum"))
    )
    resumo = resumo.merge(pares, on=["dia", "condicao"], how="left").fillna({"pares_sig_tukey": 0})
    resumo["pares_sig_tukey"] = resumo["pares_sig_tukey"].astype(int)
    resumo["prop_bandas_sig_anova"] = resumo["bandas_sig_anova"] / resumo["bandas_testadas"]
    resumo_pares = (
        df_tukey.groupby(["dia", "condicao", "genotipo_a", "genotipo_b"], as_index=False)
        .agg(bandas_avaliadas_tukey=("banda_nm", "size"),
             bandas_sig_tukey=("significativo_tukey", "sum"))
    )
    resumo_pares["prop_bandas_sig_tukey"] = (
        resumo_pares["bandas_sig_tukey"] / resumo_pares["bandas_avaliadas_tukey"]
    )

    df_anova.to_csv(SAIDA_DIR / "anova_genotipos_por_dia.csv", sep=";", index=False)
    df_tukey.to_csv(SAIDA_DIR / "tukey_genotipos_por_dia.csv", sep=";", index=False)
    resumo.to_csv(SAIDA_DIR / "anova_tukey_genotipos_resumo.csv", sep=";", index=False)
    resumo_pares.to_csv(
        SAIDA_DIR / "anova_tukey_genotipos_resumo_pares.csv", sep=";", index=False
    )

    print("ANOVA + Tukey HSD, p <= 0,05 (sem FDR entre bandas)")
    print(resumo.to_string(index=False))
    print(f"\nSalvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
