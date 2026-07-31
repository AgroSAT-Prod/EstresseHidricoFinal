#!/usr/bin/env python3
"""Teste de diferenças significativas por banda espectral e por dia.

Para cada dia de coleta (D02..D10) e cada banda do espectro pré-processado,
decide o teste pela normalidade das seis células do delineamento
(3 genótipos x 2 condições) e aplica:

- Dados normais (Shapiro-Wilk não rejeita H0 em nenhuma das 6 células):
  ANOVA de dois fatores com interação -- genotipo, condicao e genotipo:condicao.
  Somas de quadrados do tipo II (comparação de modelos aninhados), que é o
  correto para o delineamento levemente desbalanceado deste experimento.

- Dados não normais (pelo menos uma célula rejeita normalidade):
  Kruskal-Wallis sobre as 6 células combinadas (teste omnibus) e, como
  equivalentes marginais das linhas da ANOVA, Kruskal-Wallis para genotipo
  (3 grupos) e para condicao (2 grupos). O posto não tem análogo direto para
  o termo de interação -- ver nota em `INTERACAO_KW` abaixo.

- Pós-hoc de Dunn nas 15 comparações par a par entre as 6 células, apenas para
  as bandas cujo omnibus de Kruskal-Wallis sobreviveu à correção FDR.

Todos os p-valores recebem correção FDR de Benjamini-Hochberg, aplicada
dentro de cada dia e dentro de cada teste (as bandas são a família de testes).

Uso:
    python diferencas_significativas.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, norm, rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar, shapiro_por_banda  # noqa: E402
from selecao_estatistica import fdr_bh  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"

ESTAGIO_PADRAO = "normalizado"

ALPHA = 0.05

GENOTIPOS = ["BR16", "CD202", "EMB48"]
CONDICOES = ["IRRIG", "NIRRIG"]

# Kruskal-Wallis é um teste de um fator: não existe termo de interação
# rank-based equivalente ao da ANOVA. O omnibus sobre as 6 células detecta
# qualquer diferença (inclusive a que a ANOVA atribuiria à interação), e o
# pós-hoc de Dunn é quem localiza onde ela está.
INTERACAO_KW = "omnibus sobre as 6 celulas + pos-hoc de Dunn"


def dummies(rotulos: np.ndarray, niveis: list[str]) -> np.ndarray:
    """Codificação dummy com o primeiro nível como referência."""
    return np.column_stack([
        (rotulos == nivel).astype(float) for nivel in niveis[1:]
    ])


def rss_multiplo(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Soma de quadrados dos resíduos de X sobre cada coluna de Y."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    residuos = Y - X @ beta
    return np.sum(residuos**2, axis=0)


def anova_dois_fatores(
    Y: np.ndarray,
    genotipo: np.ndarray,
    condicao: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ANOVA de dois fatores com interação, vetorizada sobre as bandas.

    Todas as bandas de um mesmo dia compartilham a matriz de delineamento,
    então os modelos são resolvidos de uma vez para as (n_amostras, n_bandas).
    Usa somas de quadrados do tipo II: cada fator é testado comparando o
    modelo aditivo completo contra o modelo sem aquele fator.

    Returns:
        Mapa termo -> (estatística F, p-valor), com uma entrada por banda.
    """
    n = len(Y)
    um = np.ones((n, 1))
    g = dummies(genotipo, GENOTIPOS)
    c = dummies(condicao, CONDICOES)
    gc = np.column_stack([g[:, [i]] * c for i in range(g.shape[1])])

    X_completo = np.column_stack([um, g, c, gc])
    X_aditivo = np.column_stack([um, g, c])
    X_cond = np.column_stack([um, c])
    X_gen = np.column_stack([um, g])

    rss_completo = rss_multiplo(X_completo, Y)
    rss_aditivo = rss_multiplo(X_aditivo, Y)
    rss_cond = rss_multiplo(X_cond, Y)
    rss_gen = rss_multiplo(X_gen, Y)

    gl_residuo = n - X_completo.shape[1]
    qm_residuo = rss_completo / gl_residuo

    termos = {
        "genotipo": (rss_cond - rss_aditivo, g.shape[1]),
        "condicao": (rss_gen - rss_aditivo, c.shape[1]),
        "interacao": (rss_aditivo - rss_completo, gc.shape[1]),
    }

    resultado = {}
    for termo, (soma_quadrados, gl) in termos.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (soma_quadrados / gl) / qm_residuo
        p = f_dist.sf(F, gl, gl_residuo)
        resultado[termo] = (F, np.where(np.isfinite(F), p, np.nan))

    return resultado


def correcao_empates(Y: np.ndarray) -> np.ndarray:
    """Soma de (t^3 - t) sobre os grupos de empate, por banda."""
    Y_ord = np.sort(Y, axis=0)
    if not np.any(Y_ord[1:] == Y_ord[:-1]):
        return np.zeros(Y.shape[1])

    soma = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        _, contagens = np.unique(Y_ord[:, j], return_counts=True)
        t = contagens[contagens > 1].astype(float)
        soma[j] = np.sum(t**3 - t)
    return soma


def kruskal_vetorizado(
    postos: np.ndarray,
    grupos: list[np.ndarray],
    empates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kruskal-Wallis sobre postos já calculados, vetorizado nas bandas."""
    N = postos.shape[0]
    esperado = (N + 1) / 2

    H = np.zeros(postos.shape[1])
    for idx in grupos:
        media_postos = postos[idx].mean(axis=0)
        H += len(idx) * (media_postos - esperado) ** 2
    H *= 12.0 / (N * (N + 1))

    correcao = 1.0 - empates / (N**3 - N)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(correcao > 0, H / correcao, np.nan)

    gl = len(grupos) - 1
    return H, chi2.sf(H, gl)


def dunn_vetorizado(
    postos: np.ndarray,
    grupos: dict[str, np.ndarray],
    empates: np.ndarray,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Pós-hoc de Dunn par a par sobre os postos, vetorizado nas bandas."""
    N = postos.shape[0]
    nomes = list(grupos)
    medias = {nome: postos[idx].mean(axis=0) for nome, idx in grupos.items()}

    variancia_base = N * (N + 1) / 12.0 - empates / (12.0 * (N - 1))

    resultado = {}
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            escala = 1.0 / len(grupos[a]) + 1.0 / len(grupos[b])
            desvio = np.sqrt(np.maximum(variancia_base * escala, 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (medias[a] - medias[b]) / desvio
            p = 2.0 * norm.sf(np.abs(z))
            resultado[(a, b)] = (z, np.where(np.isfinite(z), p, np.nan))
    return resultado


def celulas_normais(Y: np.ndarray, meta_dia: pd.DataFrame) -> np.ndarray:
    """True para as bandas em que as 6 células passam no Shapiro-Wilk."""
    normal = np.ones(Y.shape[1], dtype=bool)
    for genotipo in GENOTIPOS:
        for condicao in CONDICOES:
            mask = (
                (meta_dia["genotipo"] == genotipo)
                & (meta_dia["condicao"] == condicao)
            ).to_numpy()
            if mask.sum() < 3:
                return np.zeros(Y.shape[1], dtype=bool)
            _, pvals = shapiro_por_banda(Y[mask])
            normal &= np.isfinite(pvals) & (pvals > ALPHA)
    return normal


def q_por_via(p: np.ndarray, via: np.ndarray, rotulo: str) -> np.ndarray:
    """FDR de Benjamini-Hochberg dentro do subconjunto de bandas de uma via."""
    q = np.full(len(p), np.nan)
    mask = via == rotulo
    if mask.any():
        q[mask] = np.clip(fdr_bh(p[mask]), 0, 1)
    return q


def analisar_dia(
    Y: np.ndarray,
    w: np.ndarray,
    meta_dia: pd.DataFrame,
    dia: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda ANOVA ou Kruskal-Wallis banda a banda para um dia."""
    genotipo = meta_dia["genotipo"].to_numpy()
    condicao = meta_dia["condicao"].to_numpy()

    normal = celulas_normais(Y, meta_dia)
    via = np.where(normal, "anova", "kruskal")

    n_bandas = Y.shape[1]
    p_genotipo = np.full(n_bandas, np.nan)
    p_condicao = np.full(n_bandas, np.nan)
    p_interacao = np.full(n_bandas, np.nan)
    estat_genotipo = np.full(n_bandas, np.nan)
    estat_condicao = np.full(n_bandas, np.nan)
    estat_interacao = np.full(n_bandas, np.nan)
    p_omnibus = np.full(n_bandas, np.nan)
    estat_omnibus = np.full(n_bandas, np.nan)

    if normal.any():
        anova = anova_dois_fatores(Y[:, normal], genotipo, condicao)
        estat_genotipo[normal], p_genotipo[normal] = anova["genotipo"]
        estat_condicao[normal], p_condicao[normal] = anova["condicao"]
        estat_interacao[normal], p_interacao[normal] = anova["interacao"]

    postos = rankdata(Y, axis=0)
    empates = correcao_empates(Y)

    celulas = {
        f"{g}|{c}": ((genotipo == g) & (condicao == c)).nonzero()[0]
        for g in GENOTIPOS
        for c in CONDICOES
    }
    celulas = {nome: idx for nome, idx in celulas.items() if len(idx) > 0}

    nao_normal = ~normal
    if nao_normal.any():
        grupos_gen = [(genotipo == g).nonzero()[0] for g in GENOTIPOS]
        grupos_cond = [(condicao == c).nonzero()[0] for c in CONDICOES]
        grupos_gen = [idx for idx in grupos_gen if len(idx) > 0]
        grupos_cond = [idx for idx in grupos_cond if len(idx) > 0]

        sub_postos = postos[:, nao_normal]
        sub_empates = empates[nao_normal]

        estat_genotipo[nao_normal], p_genotipo[nao_normal] = kruskal_vetorizado(
            sub_postos, grupos_gen, sub_empates
        )
        estat_condicao[nao_normal], p_condicao[nao_normal] = kruskal_vetorizado(
            sub_postos, grupos_cond, sub_empates
        )
        estat_omnibus[nao_normal], p_omnibus[nao_normal] = kruskal_vetorizado(
            sub_postos, list(celulas.values()), sub_empates
        )

    df = pd.DataFrame({
        "dia": dia,
        "banda_nm": w.astype(int),
        "n": len(Y),
        "via": via,
        "estat_genotipo": estat_genotipo,
        "p_genotipo": p_genotipo,
        "estat_condicao": estat_condicao,
        "p_condicao": p_condicao,
        "estat_interacao": estat_interacao,
        "p_interacao": p_interacao,
        "estat_omnibus": estat_omnibus,
        "p_omnibus": p_omnibus,
    })

    # FDR aplicado dentro de cada dia, cada teste e cada via -- misturar as
    # duas vias na mesma família tornaria a correção dependente da decisão
    # de normalidade, que não é o objeto do teste.
    for coluna in ["genotipo", "condicao", "interacao", "omnibus"]:
        q = np.full(n_bandas, np.nan)
        for rotulo in ("anova", "kruskal"):
            q_via = q_por_via(df[f"p_{coluna}"].to_numpy(), via, rotulo)
            q = np.where(np.isnan(q), q_via, q)
        df[f"q_{coluna}"] = q

    significativa = (
        (df["via"] == "kruskal") & (df["q_omnibus"] < ALPHA)
    ).to_numpy()

    df_dunn = pd.DataFrame()
    if significativa.any():
        dunn = dunn_vetorizado(
            postos[:, significativa], celulas, empates[significativa]
        )
        bandas_sig = w[significativa].astype(int)
        linhas = []
        for (a, b), (z, p) in dunn.items():
            linhas.append(pd.DataFrame({
                "dia": dia,
                "banda_nm": bandas_sig,
                "grupo_a": a,
                "grupo_b": b,
                "z": z,
                "p_valor": p,
            }))
        df_dunn = pd.concat(linhas, ignore_index=True)
        # A família do pós-hoc são todas as comparações do dia.
        df_dunn["q_fdr"] = np.clip(fdr_bh(df_dunn["p_valor"].to_numpy()), 0, 1)
        df_dunn["significativa"] = df_dunn["q_fdr"] < ALPHA

    return df, df_dunn


def resumir(df: pd.DataFrame, df_dunn: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dia com a contagem de bandas significativas."""
    linhas = []
    for dia, grupo in df.groupby("dia", sort=True):
        dunn_dia = df_dunn[df_dunn["dia"] == dia] if not df_dunn.empty else df_dunn
        n_anova = int((grupo["via"] == "anova").sum())
        linhas.append({
            "dia": dia,
            "n_amostras": int(grupo["n"].iloc[0]),
            "bandas": len(grupo),
            "via_anova": n_anova,
            "via_kruskal": int((grupo["via"] == "kruskal").sum()),
            "sig_genotipo": int((grupo["q_genotipo"] < ALPHA).sum()),
            "sig_condicao": int((grupo["q_condicao"] < ALPHA).sum()),
            # Vazio, e não zero, quando nenhuma banda seguiu pela ANOVA: a
            # interação não foi testada, o que é diferente de não existir.
            "sig_interacao": int((grupo["q_interacao"] < ALPHA).sum()) if n_anova else None,
            "sig_omnibus": int((grupo["q_omnibus"] < ALPHA).sum()),
            "pares_dunn": len(dunn_dia),
            "pares_dunn_sig": int(dunn_dia["significativa"].sum()) if len(dunn_dia) else 0,
        })
    return pd.DataFrame(linhas)


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio)
    print(f"{len(meta)} amostras x {len(w)} bandas\n")

    resultados = []
    resultados_dunn = []

    for dia in sorted(meta["dia"].unique()):
        mask = (meta["dia"] == dia).to_numpy()
        meta_dia = meta[mask].reset_index(drop=True)

        df, df_dunn = analisar_dia(espectro[mask], w, meta_dia, dia)
        resultados.append(df)
        if not df_dunn.empty:
            resultados_dunn.append(df_dunn)

        print(f"{dia} (n={mask.sum()}): "
              f"ANOVA em {(df['via'] == 'anova').sum()} bandas, "
              f"Kruskal-Wallis em {(df['via'] == 'kruskal').sum()}")
        print(f"    significativas (q<{ALPHA}) -- "
              f"genotipo: {(df['q_genotipo'] < ALPHA).sum():4d}  "
              f"condicao: {(df['q_condicao'] < ALPHA).sum():4d}  "
              f"interacao: {(df['q_interacao'] < ALPHA).sum():4d}  "
              f"omnibus: {(df['q_omnibus'] < ALPHA).sum():4d}")

    df_final = pd.concat(resultados, ignore_index=True)
    df_dunn_final = (
        pd.concat(resultados_dunn, ignore_index=True)
        if resultados_dunn else pd.DataFrame()
    )

    df_final.to_csv(SAIDA_DIR / "diferencas_por_banda_dia.csv", sep=";", index=False)
    if not df_dunn_final.empty:
        df_dunn_final.to_csv(SAIDA_DIR / "dunn_posthoc.csv", sep=";", index=False)

    df_resumo = resumir(df_final, df_dunn_final)
    df_resumo.to_csv(SAIDA_DIR / "diferencas_resumo.csv", sep=";", index=False)

    print(f"\nPós-hoc de Dunn: {len(df_dunn_final)} comparações, "
          f"{int(df_dunn_final['significativa'].sum()) if not df_dunn_final.empty else 0} "
          f"significativas após FDR")
    print(f"Resultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
