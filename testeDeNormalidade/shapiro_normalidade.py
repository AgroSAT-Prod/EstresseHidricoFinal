#!/usr/bin/env python3
"""Teste de normalidade Shapiro-Wilk banda a banda sobre o espectro pré-processado.

Reaplica o pipeline de `preprocessamento_espectral` em memória (recorte
400-2450 nm + jump correction + interpolação -> Savitzky-Golay -> SNV) e testa,
para cada banda, se a distribuição dos valores entre as amostras é normal.

Apenas o turno da manhã entra na análise: só 3 dos 7 dias têm coleta de tarde,
então misturar turnos injetaria variação diurna nos grupos e contaminaria o
teste de normalidade com uma mistura de duas populações (bimodalidade
artificial). Mesma restrição adotada em `analiseTemporalBandas`.

A avaliação é feita em cinco níveis de agrupamento:
1. global      - todas as amostras juntas
2. genotipo    - BR16, CD202, EMB48
3. condicao    - IRRIG, NIRRIG
4. dia         - D02..D10 (derivado de data_coleta)
5. genotipo x condicao x dia - estrato completo do delineamento

Dentro de cada grupo os p-valores das bandas recebem correção FDR
(Benjamini-Hochberg), reaproveitando `fdr_bh` de `selecao_estatistica`.
Rejeitar H0 (p/q < ALPHA) significa que a banda NÃO é normal naquele grupo.

O nível `bloco` responde à objeção de pseudorreplicação: cada célula
genótipo x condição x dia são 4 blocos x 8 varreduras da mesma parcela, e
Shapiro-Wilk pressupõe observações independentes. Agregando as 8 leituras na
média do bloco, cada valor testado passa a ser uma unidade experimental de
fato, e o que sobra de não-normalidade não pode ser atribuído à correlação
entre varreduras do mesmo alvo.

Uso:
    python shapiro_normalidade.py [recortado|suavizado|normalizado] [leitura|bloco]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDiferencaSignificativa"))

from preprocessamento import (  # noqa: E402
    ENTRADA,
    LIMITE_INF,
    LIMITE_SUP,
    preprocessar,
)
from suavizacao import savitzky_golay  # noqa: E402
from normalizacao import snv  # noqa: E402
from selecao_estatistica import fdr_bh  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"

ESTAGIOS = ("recortado", "suavizado", "normalizado")
ESTAGIO_PADRAO = "normalizado"

ALPHA = 0.05
MIN_N = 3

# Único turno presente em todos os sete dias.
TURNO = "manha"

AGRUPAMENTOS: dict[str, list[str]] = {
    "global": [],
    "genotipo": ["genotipo"],
    "condicao": ["condicao"],
    "dia": ["dia"],
    "genotipo_condicao_dia": ["genotipo", "condicao", "dia"],
}

NIVEIS = ("leitura", "bloco")
NIVEL_PADRAO = "leitura"

# Chaves da unidade experimental: as 8 varreduras dentro delas viram uma média.
CHAVES_BLOCO = ["genotipo", "condicao", "dia", "bloco"]

# No nível bloco cada dia contribui com 4 médias, então o estrato completo
# genotipo x condicao x dia cairia para n=4 -- abaixo de qualquer potência
# útil. O agrupamento mais fino que ainda se sustenta é genotipo x condicao,
# com os 7 dias juntos (n=28).
AGRUPAMENTOS_BLOCO: dict[str, list[str]] = {
    "global": [],
    "genotipo": ["genotipo"],
    "condicao": ["condicao"],
    "genotipo_condicao": ["genotipo", "condicao"],
}


def carregar(
    estagio: str,
    turno: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Lê o dataset limpo e aplica o pré-processamento até o estágio pedido.

    Args:
        estagio: até onde levar o pré-processamento.
        turno: quando dado, restringe as amostras a esse turno. Só 3 dos 7 dias
            têm coleta de tarde, então manter os dois turnos deixa D02, D03 e
            D09 com o dobro de leituras dos demais dias -- desbalanceamento que
            cai justamente sobre o fator dia.
    """
    if estagio not in ESTAGIOS:
        raise SystemExit(f"Estágio inválido: {estagio!r}. Use um de {ESTAGIOS}.")

    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    df_pre, w = preprocessar(df)
    bandas_str = [str(int(b)) for b in w]
    espectro = df_pre[bandas_str].values.astype(float)

    if estagio in ("suavizado", "normalizado"):
        espectro = savitzky_golay(espectro)
    if estagio == "normalizado":
        espectro = snv(espectro)

    meta = df_pre[[c for c in df_pre.columns if not c.isdigit()]].copy()
    meta["dia"] = meta["data_coleta"].astype(str).str.extract(r"^(D\d+)")

    if turno is not None:
        mask = (meta["turno"] == turno).to_numpy()
        if not mask.any():
            raise SystemExit(f"Nenhuma amostra no turno {turno!r}.")
        meta = meta[mask].reset_index(drop=True)
        espectro = espectro[mask]

    return meta, espectro, w


def carregar_estagios(
    turno: str | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Devolve os três estágios do pré-processamento de um único passe.

    `carregar` refaz o jump correction linha a linha a cada chamada, e esse
    laço domina o tempo de execução. Quem precisa comparar estágios -- os
    scripts de figura -- deve usar esta função em vez de chamar `carregar`
    uma vez por estágio.
    """
    df = pd.read_csv(ENTRADA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    df_pre, w = preprocessar(df)
    bandas_str = [str(int(b)) for b in w]

    recortado = df_pre[bandas_str].values.astype(float)
    suavizado = savitzky_golay(recortado)
    normalizado = snv(suavizado)

    meta = df_pre[[c for c in df_pre.columns if not c.isdigit()]].copy()
    meta["dia"] = meta["data_coleta"].astype(str).str.extract(r"^(D\d+)")

    mask = np.ones(len(meta), dtype=bool)
    if turno is not None:
        mask = (meta["turno"] == turno).to_numpy()
        meta = meta[mask].reset_index(drop=True)

    estagios = {
        "recortado": recortado[mask],
        "suavizado": suavizado[mask],
        "normalizado": normalizado[mask],
    }
    return meta, estagios, w


def agregar_por_bloco(
    meta: pd.DataFrame,
    espectro: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Colapsa as varreduras de cada parcela na média do bloco.

    Uma linha por (genotipo, condicao, dia, bloco) -- a unidade experimental do
    delineamento. Elimina a correlação entre as 8 varreduras consecutivas do
    mesmo alvo, que é o que invalidaria o Shapiro-Wilk no nível leitura.
    """
    chaves = meta[CHAVES_BLOCO].copy()
    grupos = chaves.groupby(CHAVES_BLOCO, sort=True).indices

    linhas = []
    medias = np.empty((len(grupos), espectro.shape[1]))
    for i, (valores, idx) in enumerate(sorted(grupos.items())):
        linhas.append(dict(zip(CHAVES_BLOCO, valores)))
        medias[i] = espectro[idx].mean(axis=0)

    return pd.DataFrame(linhas), medias


def shapiro_por_banda(espectro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estatística W e p-valor do Shapiro-Wilk para cada banda (coluna)."""
    n_bandas = espectro.shape[1]
    stats = np.full(n_bandas, np.nan)
    pvals = np.full(n_bandas, np.nan)

    for j in range(n_bandas):
        v = espectro[:, j]
        v = v[np.isfinite(v)]
        # Shapiro-Wilk exige n >= 3 e é indefinido para amostra constante.
        if len(v) < MIN_N or np.ptp(v) == 0:
            continue
        try:
            res = shapiro(v)
        except Exception:
            continue
        stats[j] = res.statistic
        pvals[j] = res.pvalue

    return stats, pvals


def avaliar_grupo(
    espectro: np.ndarray,
    w: np.ndarray,
) -> pd.DataFrame:
    """Roda Shapiro-Wilk em todas as bandas de um grupo e aplica FDR."""
    stats, pvals = shapiro_por_banda(espectro)
    qvals = np.clip(fdr_bh(pvals), 0, 1)

    return pd.DataFrame({
        "banda_nm": w.astype(int),
        "n": len(espectro),
        "W": stats,
        "p_valor": pvals,
        "q_fdr": qvals,
        "normal_p": pvals > ALPHA,
        "normal_q": qvals > ALPHA,
    })


def avaliar_agrupamento(
    meta: pd.DataFrame,
    espectro: np.ndarray,
    w: np.ndarray,
    chaves: list[str],
) -> pd.DataFrame:
    """Avalia normalidade em cada grupo definido por `chaves`."""
    if not chaves:
        grupos = [(("todas",), np.arange(len(meta)))]
    else:
        grupos = [
            (valores if isinstance(valores, tuple) else (valores,), idx)
            for valores, idx in meta.groupby(chaves, sort=True).indices.items()
        ]

    resultados = []
    for valores, idx in sorted(grupos, key=lambda g: g[0]):
        df_grupo = avaliar_grupo(espectro[idx], w)
        colunas = chaves if chaves else ["grupo"]
        for chave, valor in zip(reversed(colunas), reversed(valores)):
            df_grupo.insert(0, chave, valor)
        resultados.append(df_grupo)

    return pd.concat(resultados, ignore_index=True)


def resumir(df: pd.DataFrame, agrupamento: str, chaves: list[str]) -> pd.DataFrame:
    """Compacta os resultados de um agrupamento em uma linha por grupo."""
    chaves_resumo = chaves if chaves else ["grupo"]
    testadas = df[df["p_valor"].notna()]

    resumo = (
        testadas.groupby(chaves_resumo, sort=True)
        .agg(
            n_amostras=("n", "first"),
            bandas_testadas=("p_valor", "size"),
            normais_p=("normal_p", "sum"),
            normais_q=("normal_q", "sum"),
            W_mediano=("W", "median"),
        )
        .reset_index()
    )
    resumo["prop_normais_p"] = resumo["normais_p"] / resumo["bandas_testadas"]
    resumo["prop_normais_q"] = resumo["normais_q"] / resumo["bandas_testadas"]

    rotulo = resumo[chaves_resumo].astype(str).agg(" | ".join, axis=1)
    resumo = resumo.drop(columns=chaves_resumo)
    resumo.insert(0, "grupo", rotulo)
    resumo.insert(0, "agrupamento", agrupamento)

    return resumo


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO
    nivel = sys.argv[2] if len(sys.argv) > 2 else NIVEL_PADRAO
    if nivel not in NIVEIS:
        raise SystemExit(f"Nível inválido: {nivel!r}. Use um de {NIVEIS}.")

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio, turno=TURNO)
    print(f"{len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas "
          f"({LIMITE_INF}-{LIMITE_SUP} nm)")

    if nivel == "bloco":
        meta, espectro = agregar_por_bloco(meta, espectro)
        print(f"Nível bloco: {len(meta)} médias de parcela "
              f"({' x '.join(CHAVES_BLOCO)})")
    agrupamentos = AGRUPAMENTOS if nivel == "leitura" else AGRUPAMENTOS_BLOCO
    # O nível leitura mantém os nomes já referenciados pelos scripts de figura.
    base = "normalidade" if nivel == "leitura" else "normalidade_bloco"
    prefixo = f"{base}_shapiro"
    print()

    resumos = []
    df_estrato = pd.DataFrame()

    for agrupamento, chaves in agrupamentos.items():
        print(f"Shapiro-Wilk por banda - agrupamento: {agrupamento}")
        df = avaliar_agrupamento(meta, espectro, w, chaves)

        saida = SAIDA_DIR / f"{prefixo}_{agrupamento}.csv"
        df.to_csv(saida, sep=";", index=False)

        resumo = resumir(df, agrupamento, chaves)
        resumos.append(resumo)

        for _, row in resumo.iterrows():
            print(f"  {row['grupo']:<28} n={row['n_amostras']:>4}  "
                  f"normais(p): {row['prop_normais_p']:6.1%}  "
                  f"normais(q FDR): {row['prop_normais_q']:6.1%}  "
                  f"W mediano: {row['W_mediano']:.3f}")

        # Estrato mais fino do nível: é sobre ele que se conta em quantos
        # grupos cada banda sobrevive.
        if agrupamento == list(agrupamentos)[-1]:
            df_estrato = df
        print()

    df_resumo = pd.concat(resumos, ignore_index=True)
    df_resumo.to_csv(SAIDA_DIR / f"{base}_resumo.csv", sep=";", index=False)

    # Quantos dos estratos mais finos cada banda passou.
    testadas = df_estrato[df_estrato["p_valor"].notna()]
    por_banda = (
        testadas.groupby("banda_nm", sort=True)
        .agg(
            estratos=("normal_q", "size"),
            estratos_normais_p=("normal_p", "sum"),
            estratos_normais_q=("normal_q", "sum"),
            p_minimo=("p_valor", "min"),
            W_mediano=("W", "median"),
        )
        .reset_index()
    )
    por_banda["prop_estratos_normais_p"] = (
        por_banda["estratos_normais_p"] / por_banda["estratos"]
    )
    por_banda["prop_estratos_normais_q"] = (
        por_banda["estratos_normais_q"] / por_banda["estratos"]
    )
    por_banda.to_csv(SAIDA_DIR / f"{base}_por_banda.csv", sep=";", index=False)

    n_todas = int((por_banda["prop_estratos_normais_q"] == 1).sum())
    print(f"Bandas normais (q FDR) em todos os {int(por_banda['estratos'].max())} "
          f"estratos: {n_todas} de {len(por_banda)} "
          f"({n_todas / len(por_banda):.1%})")
    print(f"\nResultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
