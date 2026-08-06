#!/usr/bin/env python3
"""ANOVA + Tukey para contrastes cruzados de genotipo x condicao, por dia.

Em cada dia, as seis celulas (3 genotipos x 2 condicoes) entram em uma ANOVA
de uma via por banda. O Tukey HSD testa os 15 pares; a saida cruzada mantem
somente os seis pares entre genotipos distintos e condicoes opostas, por
exemplo BR16 irrigado x CD202 nao irrigado.
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
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"
ESTAGIO_PADRAO = "normalizado"
TURNO = "manha"
ALPHA = 0.05
GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]


def anova_uma_via(grupos: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tamanhos = np.array([len(grupo) for grupo in grupos])
    medias = np.array([grupo.mean(axis=0) for grupo in grupos])
    media_geral = np.average(medias, axis=0, weights=tamanhos)
    ss_entre = ((medias - media_geral) ** 2 * tamanhos[:, None]).sum(axis=0)
    ss_dentro = sum(((grupo - media) ** 2).sum(axis=0) for grupo, media in zip(grupos, medias))
    gl_entre, gl_dentro = len(grupos) - 1, tamanhos.sum() - len(grupos)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_valor = (ss_entre / gl_entre) / (ss_dentro / gl_dentro)
        eta2 = ss_entre / (ss_entre + ss_dentro)
    return f_valor, f_dist.sf(f_valor, gl_entre, gl_dentro), eta2


def eh_cruzado(celula_a: tuple[str, str], celula_b: tuple[str, str]) -> bool:
    """Retem somente A irrigado x B nao irrigado, com A e B distintos."""
    genotipo_a, condicao_a = celula_a
    genotipo_b, condicao_b = celula_b
    return genotipo_a != genotipo_b and condicao_a != condicao_b


def tukey_por_banda(
    grupos: list[np.ndarray], celulas: list[tuple[str, str]], bandas: np.ndarray,
    f_valor: np.ndarray, p_anova: np.ndarray, dia: str,
) -> pd.DataFrame:
    selecionadas = np.flatnonzero(p_anova <= ALPHA)
    colunas = ["dia", "banda_nm", "genotipo_a", "condicao_a", "genotipo_b", "condicao_b",
               "diferenca_media", "q_tukey", "q_critico_tukey", "significativo_tukey", "F_anova", "p_anova"]
    if not len(selecionadas):
        return pd.DataFrame(columns=colunas)

    tamanhos = np.array([len(grupo) for grupo in grupos])
    medias = np.array([grupo.mean(axis=0) for grupo in grupos])
    gl_dentro = tamanhos.sum() - len(grupos)
    ss_dentro = sum(((grupo - media) ** 2).sum(axis=0) for grupo, media in zip(grupos, medias))
    qm_dentro = ss_dentro / gl_dentro
    q_critico = studentized_range.isf(ALPHA, len(grupos), gl_dentro)

    linhas = []
    for i, j in combinations(range(len(celulas)), 2):
        diferenca = medias[i, selecionadas] - medias[j, selecionadas]
        erro = np.sqrt(qm_dentro[selecionadas] / 2 * (1 / tamanhos[i] + 1 / tamanhos[j]))
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.abs(diferenca) / erro
        a, b = celulas[i], celulas[j]
        linhas.append(pd.DataFrame({
            "dia": dia, "banda_nm": bandas[selecionadas].astype(int),
            "genotipo_a": a[0], "condicao_a": a[1],
            "genotipo_b": b[0], "condicao_b": b[1],
            "diferenca_media": diferenca, "q_tukey": q,
            "q_critico_tukey": q_critico, "significativo_tukey": q >= q_critico,
            "F_anova": f_valor[selecionadas], "p_anova": p_anova[selecionadas],
        }))
    return pd.concat(linhas, ignore_index=True)


def analisar_dia(Y: np.ndarray, meta: pd.DataFrame, bandas: np.ndarray, dia: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_dia = meta[meta["dia"] == dia]
    indices = meta_dia.index.to_numpy()
    celulas = [(genotipo, condicao) for genotipo in GENOTIPOS for condicao in CONDICOES]
    grupos = [Y[indices[(meta_dia.genotipo.to_numpy() == genotipo) & (meta_dia.condicao.to_numpy() == condicao)]]
              for genotipo, condicao in celulas]
    if any(len(grupo) < 2 for grupo in grupos):
        raise ValueError(f"Amostras insuficientes no dia {dia}.")
    f_valor, p_anova, eta2 = anova_uma_via(grupos)
    anova = pd.DataFrame({
        "dia": dia, "banda_nm": bandas.astype(int), "F_anova": f_valor,
        "p_anova": p_anova, "eta2": eta2, "significativo_anova": p_anova <= ALPHA,
    })
    return anova, tukey_por_banda(grupos, celulas, bandas, f_valor, p_anova, dia)


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    meta, Y, bandas = carregar(estagio, turno=TURNO)
    anovas, tukeys = [], []
    for dia in sorted(meta.dia.unique()):
        anova, tukey = analisar_dia(Y, meta, bandas, dia)
        anovas.append(anova)
        tukeys.append(tukey)
    df_anova = pd.concat(anovas, ignore_index=True)
    df_tukey = pd.concat(tukeys, ignore_index=True)
    cruzados = df_tukey[df_tukey.apply(lambda linha: eh_cruzado(
        (linha.genotipo_a, linha.condicao_a), (linha.genotipo_b, linha.condicao_b)), axis=1)].copy()
    resumo = (cruzados.groupby(["dia", "genotipo_a", "condicao_a", "genotipo_b", "condicao_b"], as_index=False)
              .agg(bandas_avaliadas_tukey=("banda_nm", "size"),
                   bandas_sig_tukey=("significativo_tukey", "sum")))
    resumo["prop_bandas_sig_tukey"] = resumo.bandas_sig_tukey / len(bandas)

    df_anova.to_csv(SAIDA_DIR / "anova_seis_celulas_por_dia.csv", sep=";", index=False)
    df_tukey.to_csv(SAIDA_DIR / "tukey_seis_celulas_por_dia.csv", sep=";", index=False)
    cruzados.to_csv(SAIDA_DIR / "tukey_contrastes_cruzados_por_dia.csv", sep=";", index=False)
    resumo.to_csv(SAIDA_DIR / "tukey_contrastes_cruzados_resumo.csv", sep=";", index=False)
    print("ANOVA (6 celulas) + Tukey HSD, p <= 0,05 (sem FDR entre bandas)")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
