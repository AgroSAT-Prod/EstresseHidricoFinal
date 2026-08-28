#!/usr/bin/env python3
"""Seleção de variáveis e escolha das Top 5 bandas por p-valor.

Parte das bandas representativas de `reducaoColinearidade` -- entrar aqui com
as 2051 bandas originais faria os dois métodos dividirem a importância de cada
região espectral entre dezenas de cópias quase idênticas, e nenhuma banda
sobreviveria à correção de múltiplos testes do Boruta.

Alvo: `condicao` (IRRIG vs NIRRIG), a discriminação de estresse hídrico.

Um genótipo, uma análise
------------------------
Cada genótipo é analisado de forma independente, sobre suas próprias amostras
e sobre os representantes que `reducaoColinearidade` produziu para ele. O
resultado é um Top 5 por genótipo: as regiões que respondem ao estresse não
são as mesmas nos três materiais, e um Top 5 único esconderia isso.

O teste de condição é feito separadamente para cada genótipo, usando os
p-valores de `comparacao_estresse.csv`. Portanto, uma banda é elegível quando
apresenta `p_valor <= 0,05` para aquele genótipo. Não há filtro de Spearman:
as cinco bandas são ordenadas diretamente pelo menor p-valor.

VIP (PLS-DA)
------------
PLS-DA com a condição codificada em -1/+1. O número de componentes é escolhido
por validação cruzada deixando um bloco de fora (4 blocos de campo, 4 dobras):
as 8 leituras de uma mesma planta são quase duplicatas, então dividir ao acaso
deixaria a mesma planta nos dois lados da dobra e inflaria a acurácia.

O VIP de uma banda mede sua contribuição para a variância explicada de y ao
longo de todos os componentes. Como a média dos VIP ao quadrado é 1 por
construção, VIP > 1 é o corte usual: a banda contribui acima da média.

Boruta
------
Comparação com variáveis-sombra. A cada iteração, cada banda ganha uma cópia
embaralhada (a sombra) e uma Random Forest é ajustada sobre o conjunto
duplicado; a banda marca um acerto quando sua importância supera a maior
importância entre todas as sombras daquela iteração. O número de acertos em
N iterações segue uma binomial(N, 0.5) sob a hipótese de irrelevância, e o
teste bilateral com correção de Bonferroni decide entre confirmar e rejeitar.

O ponto do Boruta é ser *all-relevant*: ele não devolve o menor conjunto que
classifica bem, mas todas as bandas com informação sobre o alvo -- que é o que
se quer de um estudo espectral, onde interessa saber quais regiões respondem
ao estresse, não apenas montar um classificador enxuto.

Top 5
-----
Para cada genótipo, seleciona as cinco bandas representativas com `p_valor <=
0,05`, em ordem crescente de p-valor. VIP, Boruta e os q-valores continuam
sendo exportados como resultados complementares, mas não definem o ranking.

Saídas (uma pasta por genótipo)
-------------------------------
- `<GENOTIPO>/pls_da_validacao.csv`   acurácia por número de componentes
- `<GENOTIPO>/vip_pls_da.csv`         VIP de cada banda representativa
- `<GENOTIPO>/boruta_resultado.csv`   acertos e decisão do Boruta
- `<GENOTIPO>/selecao_variaveis.csv`  p-valor e métricas complementares
- `<GENOTIPO>/top5_bandas.csv`        as Top 5 do genótipo
- `top5_bandas_por_genotipo.csv`      as Top 5 dos três, empilhadas

Uso:
    python selecao_variaveis.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402

SAIDA_DIR = ROOT.parent / "resultados" / "dataset_gerado"

COLINEARIDADE = ROOT.parent.parent / "reducaoColinearidade" / "dataset_gerado"
COMPARACAO_ESTRESSE = (
    ROOT.parent.parent / "testeDiferencaSignificativa" / "resultados" / "dataset_gerado"
    / "comparacao_estresse.csv"
)
TEMPORAL = (
    ROOT.parent.parent / "analiseTemporalBandas" / "dataset_gerado"
    / "temporal_por_banda.csv"
)

ESTAGIO_PADRAO = "normalizado"

# Mesmo recorte dos módulos que fornecem os critérios de significância.
TURNO = "manha"

ALVO = "condicao"
CLASSE_POSITIVA = "NIRRIG"

ALPHA = 0.05

MAX_COMPONENTES = 20
VIP_MIN = 1.0

BORUTA_ITERACOES = 100
BORUTA_ARVORES = 200
BORUTA_ALPHA = 0.05

TOP_K = 5

SEMENTE = 42


def carregar_bandas_representativas(
    w: np.ndarray,
    genotipo: str,
) -> tuple[np.ndarray, bool]:
    """Máscara das bandas representativas do genótipo na redução."""
    caminho = COLINEARIDADE / genotipo / "bandas_representativas.csv"
    if not caminho.exists():
        return np.ones(len(w), dtype=bool), False

    df = pd.read_csv(caminho, sep=";")
    bandas = set(df["banda_representante_nm"].astype(int))
    return np.array([int(b) in bandas for b in w]), True


def carregar_significancia(bandas: np.ndarray, genotipo: str) -> pd.DataFrame:
    """Menores p e q do efeito de condição para um genótipo."""
    df = pd.DataFrame({"banda_nm": bandas})

    estresse = (
        pd.read_csv(COMPARACAO_ESTRESSE, sep=";")
        if COMPARACAO_ESTRESSE.exists() else pd.DataFrame()
    )
    if not estresse.empty:
        estresse = estresse[estresse["genotipo"] == genotipo]
    if not estresse.empty:
        df["p_condicao"] = (
            estresse.groupby("banda_nm")["p_valor"].min()
            .reindex(bandas).to_numpy()
        )
        df["q_condicao"] = (
            estresse.groupby("banda_nm")["q_fdr"].min()
            .reindex(bandas).to_numpy()
        )
    else:
        df["p_condicao"] = np.nan
        df["q_condicao"] = np.nan

    if TEMPORAL.exists():
        temporal = pd.read_csv(TEMPORAL, sep=";")
        df["q_tempo"] = (
            temporal.groupby("banda_nm")["q_tempo"].min()
            .reindex(bandas).to_numpy()
        )
    else:
        df["q_tempo"] = np.nan

    df["significativa"] = (df["p_condicao"] <= ALPHA).fillna(False)
    return df


def escolher_componentes(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
) -> tuple[int, pd.DataFrame]:
    """Número de componentes do PLS por validação cruzada por bloco."""
    cv = GroupKFold(n_splits=len(np.unique(grupos)))
    dobras = list(cv.split(X, y, grupos))

    linhas = []
    for n_comp in range(1, MAX_COMPONENTES + 1):
        acertos = []
        for treino, teste in dobras:
            pls = PLSRegression(n_components=n_comp, scale=True)
            pls.fit(X[treino], y[treino])
            predito = np.sign(pls.predict(X[teste]).ravel())
            # sign(0) = 0 empataria a predição; joga para a classe negativa.
            predito[predito == 0] = -1
            acertos.append((predito == y[teste]).mean())
        linhas.append({
            "n_componentes": n_comp,
            "acuracia_cv": float(np.mean(acertos)),
            "desvio_cv": float(np.std(acertos)),
        })

    df = pd.DataFrame(linhas)
    melhor = int(df.loc[df["acuracia_cv"].idxmax(), "n_componentes"])
    return melhor, df


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """VIP de cada variável de um PLS ajustado.

    VIP_j = sqrt( p * sum_a (w_ja^2 * SSY_a) / sum_a SSY_a ), com SSY_a a
    soma de quadrados de y explicada pelo componente a.
    """
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    ssy = (q**2).ravel() * (t**2).sum(axis=0)
    w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)

    return np.sqrt(w.shape[0] * ((w_norm**2) * ssy).sum(axis=1) / ssy.sum())


def boruta(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Boruta com Random Forest e teste binomial corrigido por Bonferroni.

    Bandas confirmadas e rejeitadas saem da disputa assim que o teste decide,
    mas as rejeitadas continuam no ajuste da floresta: removê-las mudaria a
    distribuição de importâncias entre as iterações e quebraria a comparação
    com as sombras.
    """
    n, p = X.shape
    acertos = np.zeros(p, dtype=int)
    decisao = np.full(p, "tentativa", dtype=object)
    importancia = np.zeros(p)

    for iteracao in range(1, BORUTA_ITERACOES + 1):
        sombra = np.column_stack([
            X[rng.permutation(n), j] for j in range(p)
        ])
        Xs = np.hstack([X, sombra])

        rf = RandomForestClassifier(
            n_estimators=BORUTA_ARVORES,
            n_jobs=-1,
            random_state=int(rng.integers(0, 2**31 - 1)),
            class_weight="balanced",
        )
        rf.fit(Xs, y)

        imp = rf.feature_importances_
        importancia += imp[:p]
        limite = imp[p:].max()
        acertos += (imp[:p] > limite).astype(int)

        pendentes = np.flatnonzero(decisao == "tentativa")
        if len(pendentes) == 0:
            break

        # Bonferroni sobre as bandas ainda em disputa, refeito a cada
        # iteração: é a correção usada pela implementação de referência.
        corte = BORUTA_ALPHA / len(pendentes)
        for j in pendentes:
            p_valor = binomtest(int(acertos[j]), iteracao, 0.5).pvalue
            if p_valor >= corte:
                continue
            decisao[j] = "confirmada" if acertos[j] > iteracao / 2 else "rejeitada"

        if iteracao % 10 == 0 or len(pendentes) == 0:
            print(f"    iteração {iteracao:3d}: "
                  f"{int((decisao == 'confirmada').sum()):4d} confirmadas, "
                  f"{int((decisao == 'rejeitada').sum()):4d} rejeitadas, "
                  f"{int((decisao == 'tentativa').sum()):4d} em disputa")

    return pd.DataFrame({
        "iteracoes": iteracao,
        "acertos": acertos,
        "prop_acertos": acertos / iteracao,
        "importancia_media": importancia / iteracao,
        "decisao_boruta": decisao,
        "confirmada_boruta": decisao == "confirmada",
    })


def selecionar_top(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Top K bandas significativas, ordenadas pelo p-valor do genótipo."""
    candidatas = df[df["significativa"]].copy()

    if candidatas.empty:
        return candidatas

    top = candidatas.sort_values(
        ["p_condicao", "banda_nm"], kind="stable"
    ).head(TOP_K).copy()
    top.insert(0, "posicao", np.arange(1, len(top) + 1))
    return top


def analisar_genotipo(
    genotipo: str,
    meta: pd.DataFrame,
    espectro: np.ndarray,
    w: np.ndarray,
) -> pd.DataFrame:
    """Seleção completa para um genótipo. Devolve o Top 5 do genótipo."""
    saida = SAIDA_DIR / genotipo
    saida.mkdir(parents=True, exist_ok=True)

    # Reinicia o gerador por genótipo: senão o Boruta de cada um dependeria da
    # ordem em que os genótipos foram processados.
    rng = np.random.default_rng(SEMENTE)

    mask, tem_reducao = carregar_bandas_representativas(w, genotipo)
    if not tem_reducao:
        print(f"AVISO: bandas_representativas.csv de {genotipo} não "
              "encontrado -- usando todas as bandas (colinearidade não "
              "reduzida).")
    X = espectro[:, mask]
    bandas = w[mask].astype(int)
    print(f"Entrada da seleção: {X.shape[1]} bandas representativas\n")

    y = np.where(meta[ALVO].to_numpy() == CLASSE_POSITIVA, 1, -1)
    grupos = meta["bloco"].to_numpy()
    print(f"Alvo '{ALVO}': {int((y == 1).sum())} {CLASSE_POSITIVA} x "
          f"{int((y == -1).sum())} demais")

    print("\nPLS-DA -- validação cruzada por bloco:")
    n_comp, df_cv = escolher_componentes(X, y.astype(float), grupos)
    melhor = df_cv.loc[df_cv["n_componentes"] == n_comp].iloc[0]
    print(f"    {n_comp} componentes (acurácia {melhor['acuracia_cv']:.1%} "
          f"+- {melhor['desvio_cv']:.1%})")

    pls = PLSRegression(n_components=n_comp, scale=True)
    pls.fit(X, y.astype(float))
    vip = vip_scores(pls)
    print(f"    VIP > {VIP_MIN}: {int((vip > VIP_MIN).sum())} de {len(vip)} bandas")

    print("\nBoruta -- Random Forest contra variáveis-sombra:")
    df_boruta = boruta(X, y, rng)

    df_sig = carregar_significancia(bandas, genotipo)

    df = pd.DataFrame({
        "banda_nm": bandas,
        "coluna": np.arange(len(bandas)),
        "vip": vip,
        "vip_acima_de_1": vip > VIP_MIN,
    })
    df = pd.concat([df, df_boruta], axis=1)
    df["q_condicao"] = df_sig["q_condicao"].to_numpy()
    df["p_condicao"] = df_sig["p_condicao"].to_numpy()
    df["q_tempo"] = df_sig["q_tempo"].to_numpy()
    df["significativa"] = df_sig["significativa"].to_numpy()
    df["criterios_atendidos"] = (
        df["significativa"].astype(int)
        + df["vip_acima_de_1"].astype(int)
        + df["confirmada_boruta"].astype(int)
    )

    top = selecionar_top(df)

    df_cv.to_csv(saida / "pls_da_validacao.csv", sep=";", index=False)
    df[["banda_nm", "vip", "vip_acima_de_1"]].to_csv(
        saida / "vip_pls_da.csv", sep=";", index=False
    )
    df[[
        "banda_nm", "iteracoes", "acertos", "prop_acertos",
        "importancia_media", "decisao_boruta", "confirmada_boruta",
    ]].to_csv(saida / "boruta_resultado.csv", sep=";", index=False)
    df.drop(columns="coluna").to_csv(
        saida / "selecao_variaveis.csv", sep=";", index=False
    )
    if not top.empty:
        top.drop(columns="coluna").to_csv(
            saida / "top5_bandas.csv", sep=";", index=False
        )

    confirmadas = int(df["confirmada_boruta"].sum())
    print(f"\nBoruta: {confirmadas} confirmadas, "
          f"{int((df['decisao_boruta'] == 'rejeitada').sum())} rejeitadas, "
          f"{int((df['decisao_boruta'] == 'tentativa').sum())} indefinidas")
    print(f"Bandas com p_valor <= {ALPHA:.2f}: "
          f"{int(df['significativa'].sum())}")

    if top.empty:
        print("\nNenhuma banda atingiu p_valor <= 0,05.")
    else:
        print(f"\nTop {len(top)} bandas:")
        for _, row in top.iterrows():
            print(f"  {int(row['posicao'])}. {int(row['banda_nm']):4d} nm  "
                  f"VIP {row['vip']:5.2f}  "
                  f"Boruta {row['prop_acertos']:5.1%} acertos  "
                  f"p_valor {row['p_condicao']:.2e}")

    print(f"\nResultados de {genotipo} salvos em {saida}")

    if top.empty:
        return top
    top = top.drop(columns="coluna").copy()
    top.insert(0, "genotipo", genotipo)
    return top


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio, turno=TURNO)
    print(f"{len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")

    genotipos = sorted(meta["genotipo"].dropna().unique())
    print(f"Genótipos analisados de forma independente: {', '.join(genotipos)}")

    tops = []
    for genotipo in genotipos:
        mask = (meta["genotipo"] == genotipo).to_numpy()
        print(f"\n=== {genotipo} -- {int(mask.sum())} amostras ===")
        top = analisar_genotipo(
            genotipo,
            meta[mask].reset_index(drop=True),
            espectro[mask],
            w,
        )
        if not top.empty:
            tops.append(top)

    if tops:
        consolidado = SAIDA_DIR / "top5_bandas_por_genotipo.csv"
        pd.concat(tops, ignore_index=True).to_csv(
            consolidado, sep=";", index=False
        )
        print(f"\nTop 5 dos {len(tops)} genótipos em {consolidado}")
    else:
        print("\nNenhum genótipo produziu bandas com p_valor <= 0,05.")


if __name__ == "__main__":
    main()
