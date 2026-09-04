#!/usr/bin/env python3
"""Experimento multilimiar de selecao de bandas por genotipo e dia.

O fluxo usa somente o turno da manha, espectros normalizados, grupos locais de
bandas redundantes, representante pelo menor p-valor de IRRIG x NIRRIG e Top 5
de representantes significativas pelo VIP do PLS-DA. Cada limiar 0,80, 0,90 e
0,95 tem dois usos complementares: forma grupos locais com |rho| > limiar e
limita a correlacao entre bandas do Top final a |rho| < limiar. A janela dos
grupos e menor que 10 nm e o Top exige separacao de pelo menos 10 nm.

A validacao leave-one-block-out e diagnostica: a escolha supervisionada das
representantes ocorre antes das dobras. Ela permite comparar configuracoes
dentro deste experimento, mas nao estima desempenho imparcial em dados novos.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))

from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402


COMPARACAO = (
    ROOT / "testeDiferencaSignificativa" / "resultados" / "dataset_gerado"
    / "comparacao_estresse.csv"
)
DATASET_ENTRADA = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
SAIDA_PADRAO = (
    ROOT / "selecaoVariaveis" / "resultados" / "dataset_gerado"
    / "experimento_spearman_por_dia"
)

GENOTIPOS_PADRAO = ("BR16", "CD202", "EMB48")
CORRELACOES_MAXIMAS_PADRAO = (0.80, 0.90, 0.95)
ESTAGIO = "normalizado"
TURNO = "manha"
ALPHA = 0.05
TOP_K = 5
MAX_COMPONENTES = 10
JANELA_NM = 10.0
MIN_SEP_TOP_NM = 10.0
CLASSE_POSITIVA = "NIRRIG"
P_ISCLOSE_RTOL = 1e-5
P_ISCLOSE_ATOL = 1e-8

COLUNAS_ESTATISTICAS = {
    "dia", "genotipo", "banda_nm", "p_valor", "q_fdr",
    "epsilon2", "delta_cliff",
}


def sha256_arquivo(caminho: Path) -> str:
    """SHA-256 de um arquivo lido em blocos."""
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def validar_correlacoes_maximas(
    valores: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    """Normaliza e valida os limites maximos de correlacao do Top final."""
    valores = tuple(sorted(set(float(v) for v in valores)))
    if not valores or any(not 0 < v < 1 for v in valores):
        raise ValueError(
            "As correlacoes maximas devem estar estritamente entre 0 e 1."
        )
    return valores


def validar_estatisticas(
    estatisticas: pd.DataFrame,
    genotipos: tuple[str, ...],
    dias: tuple[str, ...],
    w: np.ndarray,
) -> pd.DataFrame:
    """Valida esquema, unicidade e cobertura do CSV de contraste de estresse."""
    ausentes = COLUNAS_ESTATISTICAS - set(estatisticas.columns)
    if ausentes:
        raise ValueError(
            "Colunas ausentes em comparacao_estresse.csv: "
            + ", ".join(sorted(ausentes))
        )

    estatisticas = estatisticas.copy()
    estatisticas["banda_nm"] = estatisticas["banda_nm"].astype(int)
    chaves = ["genotipo", "dia", "banda_nm"]
    duplicadas = estatisticas.duplicated(chaves, keep=False)
    if duplicadas.any():
        exemplo = estatisticas.loc[duplicadas, chaves].iloc[0].to_dict()
        raise ValueError(f"Chaves duplicadas em comparacao_estresse.csv: {exemplo}")

    bandas_esperadas = set(w.astype(int))
    erros = []
    partes = []
    for genotipo in genotipos:
        for dia in dias:
            sub = estatisticas[
                estatisticas["genotipo"].eq(genotipo)
                & estatisticas["dia"].eq(dia)
            ]
            bandas_encontradas = set(sub["banda_nm"])
            if bandas_encontradas != bandas_esperadas:
                faltantes = len(bandas_esperadas - bandas_encontradas)
                extras = len(bandas_encontradas - bandas_esperadas)
                erros.append(
                    f"{genotipo}/{dia}: {faltantes} ausentes e {extras} extras"
                )
            partes.append(sub)

    if erros:
        raise ValueError(
            "Cobertura incompleta em comparacao_estresse.csv: " + "; ".join(erros)
        )
    return pd.concat(partes, ignore_index=True)


def escolher_representantes(
    corr: np.ndarray,
    w: np.ndarray,
    grupos: np.ndarray,
    p_valores: np.ndarray,
) -> np.ndarray:
    """Uma representante por grupo: faixa do menor p, medoide e menor banda."""
    representantes = np.zeros(len(w), dtype=bool)
    for grupo in np.unique(grupos):
        idx = np.flatnonzero(grupos == grupo)
        p_grupo = p_valores[idx]
        finitos = np.isfinite(p_grupo)

        if finitos.any():
            menor_p = np.min(p_grupo[finitos])
            candidatos = idx[
                finitos & np.isclose(
                    p_grupo,
                    menor_p,
                    rtol=P_ISCLOSE_RTOL,
                    atol=P_ISCLOSE_ATOL,
                )
            ]
        else:
            candidatos = idx

        if len(candidatos) > 1:
            centralidade = np.abs(corr[np.ix_(candidatos, idx)]).mean(axis=1)
            # lexsort usa a ultima chave como primaria: maior centralidade e,
            # persistindo o empate, menor comprimento de onda.
            ordem = np.lexsort((w[candidatos], -centralidade))
            escolhido = candidatos[int(ordem[0])]
        else:
            escolhido = candidatos[0]
        representantes[escolhido] = True

    return representantes


def classificar_pls(predicoes: np.ndarray) -> np.ndarray:
    """Converte a resposta continua 0/1 do PLS-DA em classe pelo limiar 0,5."""
    return (np.asarray(predicoes).ravel() >= 0.5).astype(int)


def escolher_melhor_componente(validacao: pd.DataFrame) -> int:
    """Menor numero de componentes entre os de maior balanced accuracy."""
    melhor_score = validacao["balanced_accuracy_cv_diagnostica"].max()
    melhores = validacao[
        validacao["balanced_accuracy_cv_diagnostica"] == melhor_score
    ]
    return int(melhores["n_componentes"].min())


def escolher_componentes(
    X: np.ndarray,
    y: np.ndarray,
    blocos: np.ndarray,
) -> tuple[int, pd.DataFrame]:
    """Seleciona componentes por balanced accuracy leave-one-block-out."""
    blocos_unicos = np.unique(blocos)
    if len(blocos_unicos) < 2:
        raise ValueError("Sao necessarios ao menos dois blocos para a validacao.")
    if len(np.unique(y)) != 2:
        raise ValueError("Sao necessarias as duas condicoes para ajustar o PLS-DA.")

    cv = GroupKFold(n_splits=len(blocos_unicos))
    dobras = list(cv.split(X, y, groups=blocos))
    for treino, teste in dobras:
        if len(np.unique(y[treino])) != 2 or len(np.unique(y[teste])) != 2:
            raise ValueError("Cada dobra deve conter IRRIG e NIRRIG no treino e no teste.")

    limite = min(
        MAX_COMPONENTES,
        X.shape[1],
        min(len(treino) - 1 for treino, _ in dobras),
    )
    if limite < 1:
        raise ValueError("Dimensoes insuficientes para ajustar o PLS-DA.")

    linhas = []
    for n_componentes in range(1, limite + 1):
        scores = []
        for treino, teste in dobras:
            modelo = PLSRegression(n_components=n_componentes, scale=True)
            modelo.fit(X[treino], y[treino])
            predito = classificar_pls(modelo.predict(X[teste]))
            scores.append(balanced_accuracy_score(y[teste], predito))
        linhas.append({
            "n_componentes": n_componentes,
            "balanced_accuracy_cv_diagnostica": float(np.mean(scores)),
            "desvio_cv_diagnostico": float(np.std(scores)),
        })

    validacao = pd.DataFrame(linhas)
    return escolher_melhor_componente(validacao), validacao


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """Variable Importance in Projection de um PLS de resposta unica."""
    t = pls.x_scores_
    pesos = pls.x_weights_
    q = pls.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    denominador = ssy.sum()
    if not np.isfinite(denominador) or denominador <= 0:
        return np.full(pesos.shape[0], np.nan)
    normas = np.linalg.norm(pesos, axis=0, keepdims=True)
    pesos_normalizados = np.divide(
        pesos, normas, out=np.zeros_like(pesos), where=normas != 0
    )
    return np.sqrt(
        pesos.shape[0]
        * ((pesos_normalizados ** 2) * ssy).sum(axis=1)
        / denominador
    )


def selecionar_top_baixa_colinearidade(
    candidatas: pd.DataFrame,
    corr: np.ndarray,
    w: np.ndarray,
    correlacao_maxima: float,
    min_sep_nm: float = MIN_SEP_TOP_NM,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Top greedy por VIP, com correlacao maxima e separacao minima.

    A ordem de prioridade e VIP decrescente, p-valor crescente e comprimento de
    onda crescente. Uma candidata so entra se estiver a pelo menos ``min_sep_nm``
    e tiver |rho| estritamente menor que ``correlacao_maxima`` contra todas as
    bandas ja selecionadas.
    """
    ordenadas = candidatas[candidatas["significativa"]].sort_values(
        ["vip_pls_da", "p_valor", "banda_nm"],
        ascending=[False, True, True],
        kind="stable",
    )
    banda_para_indice = {int(banda): i for i, banda in enumerate(w.astype(int))}
    escolhidas: list[int] = []
    indices_escolhidos: list[int] = []

    for indice, linha in ordenadas.iterrows():
        banda = int(linha["banda_nm"])
        if any(abs(banda - outra) < min_sep_nm for outra in escolhidas):
            continue
        coluna = banda_para_indice[banda]
        if all(
            abs(float(corr[coluna, banda_para_indice[outra]])) < correlacao_maxima
            for outra in escolhidas
        ):
            escolhidas.append(banda)
            indices_escolhidos.append(indice)
        if len(escolhidas) == top_k:
            break

    top = ordenadas.loc[indices_escolhidos].copy()
    top.insert(0, "posicao", np.arange(1, len(top) + 1))

    rho_max, distancia_min = [], []
    for banda in escolhidas:
        outras = [outra for outra in escolhidas if outra != banda]
        if not outras:
            rho_max.append(np.nan)
            distancia_min.append(np.nan)
            continue
        coluna = banda_para_indice[banda]
        rho_max.append(max(
            abs(float(corr[coluna, banda_para_indice[outra]]))
            for outra in outras
        ))
        distancia_min.append(min(abs(banda - outra) for outra in outras))
    top["rho_abs_max_outro_top"] = rho_max
    top["distancia_min_outro_top_nm"] = distancia_min
    return top


def metadados_grupos(
    w: np.ndarray,
    grupos: np.ndarray,
    representantes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expande limites, tamanho e representante do grupo para cada banda."""
    inicio = np.zeros(len(w), dtype=int)
    fim = np.zeros(len(w), dtype=int)
    tamanho = np.zeros(len(w), dtype=int)
    banda_representante = np.zeros(len(w), dtype=int)
    bandas = w.astype(int)

    for grupo in np.unique(grupos):
        idx = np.flatnonzero(grupos == grupo)
        rep = idx[representantes[idx]][0]
        inicio[idx] = bandas[idx].min()
        fim[idx] = bandas[idx].max()
        tamanho[idx] = len(idx)
        banda_representante[idx] = bandas[rep]
    return inicio, fim, tamanho, banda_representante


def analisar_cenario(
    genotipo: str,
    dia: str,
    correlacao_maxima: float,
    meta: pd.DataFrame,
    espectro: np.ndarray,
    w: np.ndarray,
    corr: np.ndarray,
    estatisticas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Executa um genotipo x dia x limiar e devolve tabelas sem gravar arquivos."""
    bandas = w.astype(int)
    estat = estatisticas.set_index("banda_nm").reindex(bandas)
    if estat[list(COLUNAS_ESTATISTICAS - {"dia", "genotipo", "banda_nm"})].isna().all(axis=1).any():
        raise ValueError(f"Estatisticas ausentes para alguma banda em {genotipo}/{dia}.")

    p_valores = estat["p_valor"].to_numpy(dtype=float)
    grupos = agrupar(
        corr, w, limiar_r=correlacao_maxima, janela_nm=JANELA_NM
    )
    representantes = escolher_representantes(corr, w, grupos, p_valores)
    X_reduzido = espectro[:, representantes]
    y = meta["condicao"].eq(CLASSE_POSITIVA).astype(int).to_numpy()
    blocos = meta["bloco"].astype(str).to_numpy()

    n_componentes, validacao = escolher_componentes(X_reduzido, y, blocos)
    modelo = PLSRegression(n_components=n_componentes, scale=True)
    modelo.fit(X_reduzido, y)
    vip_representantes = vip_scores(modelo)

    vip = np.full(len(w), np.nan)
    vip[representantes] = vip_representantes
    significativa = estat["q_fdr"].to_numpy(dtype=float) < ALPHA

    candidatas = pd.DataFrame({
        "banda_nm": bandas[representantes],
        "p_valor": estat.loc[bandas[representantes], "p_valor"].to_numpy(),
        "q_fdr": estat.loc[bandas[representantes], "q_fdr"].to_numpy(),
        "epsilon2": estat.loc[bandas[representantes], "epsilon2"].to_numpy(),
        "delta_cliff": estat.loc[bandas[representantes], "delta_cliff"].to_numpy(),
        "grupo_spearman": grupos[representantes],
        "vip_pls_da": vip_representantes,
    })
    candidatas["significativa"] = candidatas["q_fdr"] < ALPHA
    candidatas["vip_acima_de_1"] = candidatas["vip_pls_da"] > 1.0
    candidatas["abs_delta_cliff"] = candidatas["delta_cliff"].abs()
    top = selecionar_top_baixa_colinearidade(
        candidatas, corr, w,
        correlacao_maxima=correlacao_maxima,
        min_sep_nm=MIN_SEP_TOP_NM,
        top_k=TOP_K,
    )
    top.insert(0, "correlacao_maxima_aceita_top", correlacao_maxima)
    top.insert(0, "limiar_agrupamento_abs_rho", correlacao_maxima)
    top.insert(0, "dia", dia)
    top.insert(0, "genotipo", genotipo)

    posicao_top = np.full(len(w), np.nan)
    banda_para_indice = {banda: i for i, banda in enumerate(bandas)}
    for _, linha in top.iterrows():
        posicao_top[banda_para_indice[int(linha["banda_nm"])]] = int(linha["posicao"])

    inicio, fim, tamanho, banda_representante = metadados_grupos(
        w, grupos, representantes
    )
    detalhe = pd.DataFrame({
        "genotipo": genotipo,
        "dia": dia,
        "limiar_agrupamento_abs_rho": correlacao_maxima,
        "correlacao_maxima_aceita_top": correlacao_maxima,
        "banda_nm": bandas,
        "grupo_spearman": grupos,
        "inicio_grupo_nm": inicio,
        "fim_grupo_nm": fim,
        "tamanho_grupo": tamanho,
        "banda_representante_nm": banda_representante,
        "representante": representantes,
        "p_valor": estat["p_valor"].to_numpy(),
        "q_fdr": estat["q_fdr"].to_numpy(),
        "epsilon2": estat["epsilon2"].to_numpy(),
        "delta_cliff": estat["delta_cliff"].to_numpy(),
        "significativa": significativa,
        "vip_pls_da": vip,
        "vip_acima_de_1": np.isfinite(vip) & (vip > 1.0),
        "posicao_top5": posicao_top,
    })

    validacao.insert(0, "correlacao_maxima_aceita_top", correlacao_maxima)
    validacao.insert(0, "limiar_agrupamento_abs_rho", correlacao_maxima)
    validacao.insert(0, "dia", dia)
    validacao.insert(0, "genotipo", genotipo)
    melhor = validacao[validacao["n_componentes"] == n_componentes].iloc[0]
    n_grupos = int(grupos.max()) + 1
    resumo = {
        "genotipo": genotipo,
        "dia": dia,
        "limiar_agrupamento_abs_rho": correlacao_maxima,
        "correlacao_maxima_aceita_top": correlacao_maxima,
        "turno": TURNO,
        "estagio": ESTAGIO,
        "amostras": len(meta),
        "bandas_originais": len(w),
        "grupos_spearman": n_grupos,
        "reducao": 1.0 - n_grupos / len(w),
        "janela_nm": JANELA_NM,
        "distancia_minima_top_nm": MIN_SEP_TOP_NM,
        "n_componentes_pls_da": n_componentes,
        "balanced_accuracy_cv_diagnostica": float(
            melhor["balanced_accuracy_cv_diagnostica"]
        ),
        "desvio_cv_diagnostico": float(melhor["desvio_cv_diagnostico"]),
        "bandas_significativas_representativas": int(
            (representantes & significativa).sum()
        ),
        "top5_encontradas": len(top),
        "rho_abs_max_observado_top": (
            float(top["rho_abs_max_outro_top"].max()) if len(top) > 1 else np.nan
        ),
        "distancia_min_observada_top_nm": (
            float(top["distancia_min_outro_top_nm"].min()) if len(top) > 1 else np.nan
        ),
        "status": (
            "ok" if len(top) == TOP_K
            else "sem_bandas_significativas"
            if not (representantes & significativa).any()
            else "menos_de_5_selecionadas"
        ),
    }
    return detalhe, top, validacao, resumo


def comparar_correlacoes_maximas(
    top5: pd.DataFrame,
    genotipos: tuple[str, ...],
    dias: tuple[str, ...],
    correlacoes_maximas: tuple[float, ...],
) -> pd.DataFrame:
    """Jaccard dos Top 5 para cada par de limites maximos de correlacao."""
    colunas = [
        "genotipo", "dia", "correlacao_maxima_a", "correlacao_maxima_b",
        "n_top_a", "n_top_b",
        "n_intersecao", "n_uniao", "jaccard", "bandas_intersecao",
        "bandas_so_a", "bandas_so_b",
    ]
    linhas = []
    for genotipo in genotipos:
        for dia in dias:
            sub = top5[top5["genotipo"].eq(genotipo) & top5["dia"].eq(dia)]
            for limite_a, limite_b in itertools.combinations(correlacoes_maximas, 2):
                a = set(
                    sub[np.isclose(sub["correlacao_maxima_aceita_top"], limite_a)][
                        "banda_nm"
                    ].astype(int)
                )
                b = set(
                    sub[np.isclose(sub["correlacao_maxima_aceita_top"], limite_b)][
                        "banda_nm"
                    ].astype(int)
                )
                intersecao = a & b
                uniao = a | b
                linhas.append({
                    "genotipo": genotipo,
                    "dia": dia,
                    "correlacao_maxima_a": limite_a,
                    "correlacao_maxima_b": limite_b,
                    "n_top_a": len(a),
                    "n_top_b": len(b),
                    "n_intersecao": len(intersecao),
                    "n_uniao": len(uniao),
                    "jaccard": len(intersecao) / len(uniao) if uniao else 1.0,
                    "bandas_intersecao": ",".join(map(str, sorted(intersecao))),
                    "bandas_so_a": ",".join(map(str, sorted(a - b))),
                    "bandas_so_b": ",".join(map(str, sorted(b - a))),
                })
    return pd.DataFrame(linhas, columns=colunas)


def executar_experimento(
    meta: pd.DataFrame,
    espectro: np.ndarray,
    w: np.ndarray,
    estatisticas: pd.DataFrame,
    genotipos: tuple[str, ...],
    correlacoes_maximas: tuple[float, ...],
) -> dict[str, pd.DataFrame]:
    """Executa todos os cenarios e devolve as cinco tabelas analiticas."""
    dias = tuple(sorted(meta["dia"].dropna().astype(str).unique()))
    estatisticas = validar_estatisticas(estatisticas, genotipos, dias, w)
    detalhes, tops, validacoes, resumos = [], [], [], []

    for genotipo in genotipos:
        mask_genotipo = meta["genotipo"].eq(genotipo).to_numpy()
        if not mask_genotipo.any():
            raise ValueError(f"Genotipo sem amostras no turno da manha: {genotipo}")
        meta_genotipo = meta.loc[mask_genotipo].reset_index(drop=True)
        espectro_genotipo = espectro[mask_genotipo]

        for dia in dias:
            mask_dia = meta_genotipo["dia"].eq(dia).to_numpy()
            meta_dia = meta_genotipo.loc[mask_dia].reset_index(drop=True)
            espectro_dia = espectro_genotipo[mask_dia]
            estat_dia = estatisticas[
                estatisticas["genotipo"].eq(genotipo)
                & estatisticas["dia"].eq(dia)
            ]
            corr = spearman_matriz(espectro_dia)

            for correlacao_maxima in correlacoes_maximas:
                detalhe, top, validacao, resumo = analisar_cenario(
                    genotipo, dia, correlacao_maxima, meta_dia, espectro_dia, w,
                    corr, estat_dia,
                )
                detalhes.append(detalhe)
                tops.append(top)
                validacoes.append(validacao)
                resumos.append(resumo)
                print(
                    f"{genotipo}/{dia} |rho|>{correlacao_maxima:.2f} nos "
                    f"grupos e |rho|<{correlacao_maxima:.2f} no Top: "
                    f"{resumo['grupos_spearman']} representantes; "
                    f"PLS-DA {resumo['n_componentes_pls_da']} comp.; "
                    f"BAC diagnostica "
                    f"{resumo['balanced_accuracy_cv_diagnostica']:.3f}; "
                    f"top {resumo['top5_encontradas']}"
                )

    detalhe_df = pd.concat(detalhes, ignore_index=True)
    top_df = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
    validacao_df = pd.concat(validacoes, ignore_index=True)
    resumo_df = pd.DataFrame(resumos)
    comparacao_df = comparar_correlacoes_maximas(
        top_df, genotipos, dias, correlacoes_maximas
    )
    return {
        "resumo_cenarios.csv": resumo_df,
        "bandas_detalhadas.csv": detalhe_df,
        "top5_bandas.csv": top_df,
        "pls_da_validacao.csv": validacao_df,
        "comparacao_limiares.csv": comparacao_df,
    }


def criar_configuracao(
    genotipos: tuple[str, ...],
    dias: tuple[str, ...],
    correlacoes_maximas: tuple[float, ...],
) -> pd.DataFrame:
    """Uma linha com configuracao, versoes e proveniencia das entradas."""
    return pd.DataFrame([{
        "executado_em_utc": datetime.now(timezone.utc).isoformat(),
        "genotipos": ",".join(genotipos),
        "dias": ",".join(dias),
        "limiares_spearman_abs_rho": ",".join(
            f"{v:.2f}" for v in correlacoes_maximas
        ),
        "comparador_agrupamento": ">",
        "janela_nm": JANELA_NM,
        "comparador_janela": "<",
        "correlacoes_maximas_aceitas_top": ",".join(
            f"{v:.2f}" for v in correlacoes_maximas
        ),
        "comparador_correlacao_top": "<",
        "distancia_minima_top_nm": MIN_SEP_TOP_NM,
        "comparador_distancia_top": ">=",
        "desempate_p_representante": "np.isclose",
        "p_isclose_rtol": P_ISCLOSE_RTOL,
        "p_isclose_atol": P_ISCLOSE_ATOL,
        "estagio": ESTAGIO,
        "turno": TURNO,
        "alpha_fdr": ALPHA,
        "comparador_fdr": "<",
        "top_k": TOP_K,
        "max_componentes_pls_da": MAX_COMPONENTES,
        "classe_positiva": CLASSE_POSITIVA,
        "limiar_predicao_pls_da": 0.5,
        "validacao": "GroupKFold leave-one-block-out",
        "grupo_validacao": "bloco",
        "natureza_validacao": "diagnostica; selecao supervisionada anterior as dobras",
        "fonte_estatisticas": COMPARACAO.relative_to(ROOT).as_posix(),
        "sha256_estatisticas": sha256_arquivo(COMPARACAO),
        "fonte_espectros": DATASET_ENTRADA.relative_to(ROOT).as_posix(),
        "sha256_espectros": sha256_arquivo(DATASET_ENTRADA),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }])


def salvar_tabelas_atomicas(tabelas: dict[str, pd.DataFrame], saida: Path) -> None:
    """Grava todas as tabelas temporariamente antes de substituir as finais."""
    saida = saida.resolve()
    saida.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tmp_experimento_spearman_", dir=saida.parent
    ) as temporario:
        pasta_temporaria = Path(temporario)
        for nome, tabela in tabelas.items():
            tabela.to_csv(pasta_temporaria / nome, sep=";", index=False)

        saida.mkdir(parents=True, exist_ok=True)
        for nome in tabelas:
            os.replace(pasta_temporaria / nome, saida / nome)


def executar_e_salvar(
    genotipos: tuple[str, ...] = GENOTIPOS_PADRAO,
    correlacoes_maximas: tuple[float, ...] = CORRELACOES_MAXIMAS_PADRAO,
    saida: Path = SAIDA_PADRAO,
) -> dict[str, pd.DataFrame]:
    """Carrega entradas, executa os cenarios e salva as seis tabelas."""
    correlacoes_maximas = validar_correlacoes_maximas(correlacoes_maximas)
    if not COMPARACAO.exists():
        raise FileNotFoundError(f"Arquivo estatistico nao encontrado: {COMPARACAO}")
    if not DATASET_ENTRADA.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_ENTRADA}")

    print(f"Carregando espectros {ESTAGIO!r} do turno {TURNO!r}...")
    meta, espectro, w = carregar(ESTAGIO, turno=TURNO)
    dias = tuple(sorted(meta["dia"].dropna().astype(str).unique()))
    estatisticas = pd.read_csv(COMPARACAO, sep=";")
    print(
        f"{len(meta)} amostras x {len(w)} bandas; "
        f"{len(genotipos)} genotipos x {len(dias)} dias x "
        f"{len(correlacoes_maximas)} limiares de Spearman\n"
    )

    tabelas = executar_experimento(
        meta, espectro, w, estatisticas, genotipos, correlacoes_maximas
    )
    tabelas = {
        "configuracao.csv": criar_configuracao(
            genotipos, dias, correlacoes_maximas
        ),
        **tabelas,
    }
    salvar_tabelas_atomicas(tabelas, saida)
    print(f"\nSeis tabelas consolidadas salvas em {saida.resolve()}")
    return tabelas


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Selecao Spearman + PLS-DA por genotipo e dia, somente manha, "
            "variando o limiar usado nos grupos e no Top."
        )
    )
    parser.add_argument(
        "--genotipos", nargs="+", choices=GENOTIPOS_PADRAO,
        default=list(GENOTIPOS_PADRAO),
        help="Genotipos a processar (padrao: os tres).",
    )
    parser.add_argument(
        "--limiares", "--correlacoes-maximas",
        dest="correlacoes_maximas", nargs="+", type=float, metavar="LIMIAR",
        default=list(CORRELACOES_MAXIMAS_PADRAO),
        help=(
            "Limiares de Spearman: grupos usam |rho| > limiar e o Top usa "
            "|rho| < limiar (padrao: 0.80 0.90 0.95)."
        ),
    )
    parser.add_argument(
        "--saida", type=Path, default=SAIDA_PADRAO,
        help="Pasta das seis tabelas consolidadas.",
    )
    return parser


def main() -> None:
    args = criar_parser().parse_args()
    executar_e_salvar(
        genotipos=tuple(dict.fromkeys(args.genotipos)),
        correlacoes_maximas=validar_correlacoes_maximas(
            args.correlacoes_maximas
        ),
        saida=args.saida,
    )


if __name__ == "__main__":
    main()
