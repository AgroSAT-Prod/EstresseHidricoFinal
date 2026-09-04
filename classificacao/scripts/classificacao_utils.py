#!/usr/bin/env python3
"""Funcoes compartilhadas para classificacao com bandas selecionadas."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

SCRIPTS_DIR = Path(__file__).resolve().parent
CLASSIFICACAO_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = CLASSIFICACAO_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402

SAIDA_DIR = CLASSIFICACAO_DIR / "outputs"
# A selecao de variaveis passou a rodar por genotipo, entao nao existe mais um
# Top 5 do pool inteiro. O modo `agrupado` usa o Top 5 de um genotipo de
# referencia -- CD202, o de resposta mais forte ao estresse -- treinando sobre
# todas as amostras.
GENOTIPO_REFERENCIA = "CD202"
TOP5_GERAL_PATH = (
    PROJECT_ROOT / "selecaoVariaveis" / "resultados" / "dataset_gerado" / GENOTIPO_REFERENCIA
    / "top5_bandas.csv"
)
SELECAO_DIR = PROJECT_ROOT / "selecaoVariaveis" / "resultados" / "dataset_gerado"
# Fonte antiga das bandas por genotipo: o ranking so por q_fdr de
# `testeDiferencaSignificativa/testar_bandas_por_genotipo.py`. Fica como
# fallback -- a fonte corrente e a Top 5 de `selecaoVariaveis`, que aplica os
# quatro criterios (significancia, colinearidade, VIP e Boruta).
BANDAS_GENOTIPO_DIR = PROJECT_ROOT / "dataset" / "dataset_gerado"

ESTAGIO_PADRAO = "normalizado"
MODO_PADRAO = "agrupado"
MODOS = ("agrupado", "por_genotipo", "ambos")
ALVO = "condicao"
CLASSE_POSITIVA = "NIRRIG"
K_FOLDS = 5
TOP_K = 5
SEMENTE = 42
GENOTIPOS = ("BR16", "CD202", "EMB48")
TURNO_AVALIACAO = "manha"
GRUPO_VALIDACAO = "nomenclaura"
ESTRATEGIA_VALIDACAO = "StratifiedGroupKFold"


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """PLS-DA binario usando PLSRegression e limiar em 0.5."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.encoder_ = LabelEncoder()
        y_num = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_

        if len(self.classes_) != 2:
            raise ValueError("PLSDAClassifier suporta apenas classificacao binaria.")

        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.n_components_ = n_comp
        self.modelo_ = PLSRegression(n_components=n_comp, scale=True)
        self.modelo_.fit(X, y_num.astype(float))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.modelo_.predict(X).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = (self.decision_function(X) >= 0.5).astype(int)
        return self.encoder_.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-(score - 0.5)))
        return np.column_stack([1.0 - prob_pos, prob_pos])


def criar_parser(descricao: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument(
        "--modo",
        choices=MODOS,
        default=MODO_PADRAO,
        help="Usa bandas gerais, bandas por genotipo ou ambos os cenarios.",
    )
    parser.add_argument(
        "--estagio",
        choices=("recortado", "suavizado", "normalizado"),
        default=ESTAGIO_PADRAO,
        help="Estagio de preprocessamento espectral.",
    )
    parser.add_argument(
        "--salvar-csv",
        dest="salvar_csv",
        action="store_true",
        default=True,
        help="Salva os CSVs de saida. Padrao: ligado.",
    )
    parser.add_argument(
        "--sem-csv",
        dest="salvar_csv",
        action="store_false",
        help="Executa a avaliacao sem criar CSVs.",
    )
    return parser


def carregar_bandas_gerais() -> list[int]:
    if not TOP5_GERAL_PATH.exists():
        raise FileNotFoundError(f"Arquivo de bandas gerais nao encontrado: {TOP5_GERAL_PATH}")

    top5 = pd.read_csv(TOP5_GERAL_PATH, sep=";")
    return top5.sort_values("posicao")["banda_nm"].astype(int).head(TOP_K).tolist()


def carregar_bandas_por_genotipo(genotipo: str) -> list[int]:
    """Top 5 do genotipo, pelos quatro criterios da selecao de variaveis."""
    caminho = SELECAO_DIR / genotipo / "top5_bandas.csv"
    if caminho.exists():
        df = pd.read_csv(caminho, sep=";")
        return df.sort_values("posicao")["banda_nm"].astype(int).head(TOP_K).tolist()

    antigo = BANDAS_GENOTIPO_DIR / f"bandas_{genotipo}.csv"
    if not antigo.exists():
        raise FileNotFoundError(
            f"Bandas de {genotipo} nao encontradas em {caminho} nem em {antigo}."
        )

    print(f"AVISO: {caminho} nao encontrado -- usando o ranking so por q_fdr "
          f"de {antigo.name}.")
    df = pd.read_csv(antigo, sep=";")
    return df.sort_values("rank")["banda_nm"].astype(int).head(TOP_K).tolist()


def selecionar_colunas(espectro: np.ndarray, w: np.ndarray, bandas: list[int]) -> np.ndarray:
    banda_para_coluna = {int(banda): i for i, banda in enumerate(w.astype(int))}
    ausentes = [banda for banda in bandas if banda not in banda_para_coluna]
    if ausentes:
        raise ValueError(f"Bandas ausentes no dataset preprocessado: {ausentes}")

    indices = [banda_para_coluna[banda] for banda in bandas]
    return espectro[:, indices].astype(float)


def montar_cenarios(modo: str, estagio: str) -> list[dict[str, object]]:
    meta, espectro, w = carregar(estagio, turno=TURNO_AVALIACAO)
    cenarios: list[dict[str, object]] = []

    if modo in ("agrupado", "ambos"):
        bandas = carregar_bandas_gerais()
        cenarios.append({
            "modo": "agrupado",
            "genotipo": "todos",
            "meta": meta.copy(),
            "X": selecionar_colunas(espectro, w, bandas),
            "y": meta[ALVO].astype(str).to_numpy(),
            "bandas": bandas,
        })

    if modo in ("por_genotipo", "ambos"):
        for genotipo in GENOTIPOS:
            mask = meta["genotipo"].astype(str).eq(genotipo).to_numpy()
            bandas = carregar_bandas_por_genotipo(genotipo)
            cenarios.append({
                "modo": "por_genotipo",
                "genotipo": genotipo,
                "meta": meta.loc[mask].copy(),
                "X": selecionar_colunas(espectro[mask], w, bandas),
                "y": meta.loc[mask, ALVO].astype(str).to_numpy(),
                "bandas": bandas,
            })

    return cenarios


def obter_score_auc(modelo: Pipeline, X: np.ndarray, classe_positiva: str = CLASSE_POSITIVA) -> np.ndarray:
    classes = list(modelo.classes_)
    pos_idx = classes.index(classe_positiva)

    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, pos_idx]

    if hasattr(modelo, "decision_function"):
        score = modelo.decision_function(X)
        return score[:, pos_idx] if np.ndim(score) > 1 else score

    raise AttributeError("Modelo sem predict_proba ou decision_function para AUC-ROC.")


def calcular_metricas(
    nome: str,
    modo: str,
    genotipo: str,
    fold: int | str,
    y_real: np.ndarray,
    y_pred: np.ndarray,
    score_auc: np.ndarray,
) -> dict[str, float | int | str]:
    y_bin = (y_real == CLASSE_POSITIVA).astype(int)
    return {
        "modelo": nome,
        "modo": modo,
        "genotipo": genotipo,
        "fold": fold,
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0),
        "recall": recall_score(y_real, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0),
        "kappa": cohen_kappa_score(y_real, y_pred),
        "auc_roc": roc_auc_score(y_bin, score_auc),
    }


def saida_do_cenario(nome: str, modo: str, genotipo: str) -> Path:
    if modo == "agrupado":
        return SAIDA_DIR / "agrupado" / nome
    return SAIDA_DIR / "por_genotipo" / genotipo / nome


def adicionar_contexto(df: pd.DataFrame, nome: str, modo: str, genotipo: str, fold: int) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "fold", fold)
    df.insert(0, "genotipo", genotipo)
    df.insert(0, "modo", modo)
    df.insert(0, "modelo", nome)
    return df


def montar_fold_assignments(
    nome: str,
    modo: str,
    genotipo: str,
    fold: int,
    meta: pd.DataFrame,
    treino_idx: np.ndarray,
    teste_idx: np.ndarray,
) -> pd.DataFrame:
    colunas = [
        "nomenclaura",
        "bloco",
        "genotipo",
        "condicao",
        "data_coleta",
        "turno",
        "dia",
    ]

    partes = []
    for split, indices in [("treino", treino_idx), ("teste", teste_idx)]:
        df_split = meta.iloc[indices][colunas].copy()
        df_split.insert(0, "indice_amostra", meta.index.to_numpy()[indices])
        df_split.insert(0, "split", split)
        df_split.insert(0, "fold", fold)
        df_split.insert(0, "genotipo_avaliado", genotipo)
        df_split.insert(0, "modo", modo)
        df_split.insert(0, "modelo", nome)
        partes.append(df_split)

    return pd.concat(partes, ignore_index=True)


def avaliar_cenario(
    nome: str,
    modelo_base: Pipeline,
    cenario: dict[str, object],
    estagio: str,
    salvar_csv: bool,
    extras_fn: Callable[[Pipeline, dict[str, object]], dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    modo = str(cenario["modo"])
    genotipo = str(cenario["genotipo"])
    meta = cenario["meta"]
    X = cenario["X"]
    y = cenario["y"]
    bandas = cenario["bandas"]

    if len(np.unique(y)) != 2:
        raise ValueError(f"Cenario {modo}/{genotipo} nao tem duas classes em {ALVO}.")

    if GRUPO_VALIDACAO not in meta.columns:
        raise ValueError(f"Coluna de grupo ausente para validacao: {GRUPO_VALIDACAO}")

    grupos = meta[GRUPO_VALIDACAO].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    metricas = []
    predicoes = []
    matrizes = []
    relatorios = []
    curvas = []
    fold_assignments = []
    extras_acumulados: dict[str, list[pd.DataFrame]] = {}

    for fold, (treino_idx, teste_idx) in enumerate(cv.split(X, y, groups=grupos), start=1):
        fold_assignments.append(
            montar_fold_assignments(nome, modo, genotipo, fold, meta, treino_idx, teste_idx)
        )

        modelo = clone(modelo_base)
        modelo.fit(X[treino_idx], y[treino_idx])

        y_pred = modelo.predict(X[teste_idx])
        score_auc = obter_score_auc(modelo, X[teste_idx])

        metricas.append(calcular_metricas(nome, modo, genotipo, fold, y[teste_idx], y_pred, score_auc))

        df_pred = meta.iloc[teste_idx].copy()
        df_pred.insert(0, "indice_amostra", meta.index.to_numpy()[teste_idx])
        df_pred.insert(0, "fold", fold)
        df_pred.insert(0, "genotipo_avaliado", genotipo)
        df_pred.insert(0, "modo", modo)
        df_pred.insert(0, "modelo", nome)
        df_pred["classe_real"] = y[teste_idx]
        df_pred["classe_predita"] = y_pred
        df_pred[f"score_{CLASSE_POSITIVA}"] = score_auc
        df_pred["acerto"] = y_pred == y[teste_idx]
        for pos, banda in enumerate(bandas):
            df_pred[str(banda)] = X[teste_idx, pos]
        predicoes.append(df_pred)

        labels = sorted(np.unique(np.concatenate([y[teste_idx], y_pred])))
        cm = confusion_matrix(y[teste_idx], y_pred, labels=labels)
        df_cm = (
            pd.DataFrame(cm, index=labels, columns=labels)
            .rename_axis("real")
            .reset_index()
            .melt(id_vars="real", var_name="predito", value_name="n")
        )
        matrizes.append(adicionar_contexto(df_cm, nome, modo, genotipo, fold))

        report = classification_report(y[teste_idx], y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).T.reset_index(names="classe")
        relatorios.append(adicionar_contexto(df_report, nome, modo, genotipo, fold))

        y_bin = (y[teste_idx] == CLASSE_POSITIVA).astype(int)
        fpr, tpr, thresholds = roc_curve(y_bin, score_auc)
        df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        curvas.append(adicionar_contexto(df_roc, nome, modo, genotipo, fold))

        if extras_fn:
            dados_fold = {
                "modo": modo,
                "genotipo": genotipo,
                "fold": fold,
                "meta": meta,
                "bandas": bandas,
                "X_treino": X[treino_idx],
                "X_teste": X[teste_idx],
                "y_treino": y[treino_idx],
                "y_teste": y[teste_idx],
            }
            for arquivo, df_extra in extras_fn(modelo, dados_fold).items():
                extras_acumulados.setdefault(arquivo, []).append(
                    adicionar_contexto(df_extra, nome, modo, genotipo, fold)
                )

    df_metricas = pd.DataFrame(metricas)
    df_resumo = (
        df_metricas
        .drop(columns=["fold"])
        .groupby(["modelo", "modo", "genotipo"], as_index=False)
        .agg(
            accuracy_media=("accuracy", "mean"),
            accuracy_desvio=("accuracy", "std"),
            precision_media=("precision", "mean"),
            precision_desvio=("precision", "std"),
            recall_media=("recall", "mean"),
            recall_desvio=("recall", "std"),
            f1_score_media=("f1_score", "mean"),
            f1_score_desvio=("f1_score", "std"),
            kappa_media=("kappa", "mean"),
            kappa_desvio=("kappa", "std"),
            auc_roc_media=("auc_roc", "mean"),
            auc_roc_desvio=("auc_roc", "std"),
        )
    )

    if salvar_csv:
        saida = saida_do_cenario(nome, modo, genotipo)
        saida.mkdir(parents=True, exist_ok=True)
        df_metricas.to_csv(saida / "metricas_folds.csv", sep=";", index=False)
        df_resumo.to_csv(saida / "metricas_resumo.csv", sep=";", index=False)
        pd.concat(predicoes, ignore_index=True).to_csv(saida / "predicoes.csv", sep=";", index=False)
        pd.concat(matrizes, ignore_index=True).to_csv(saida / "matriz_confusao_folds.csv", sep=";", index=False)
        pd.concat(relatorios, ignore_index=True).to_csv(saida / "relatorio_classificacao_folds.csv", sep=";", index=False)
        pd.concat(curvas, ignore_index=True).to_csv(saida / "curva_roc_folds.csv", sep=";", index=False)
        pd.concat(fold_assignments, ignore_index=True).to_csv(saida / "fold_assignments.csv", sep=";", index=False)

        pd.DataFrame([{
            "modelo": nome,
            "modo": modo,
            "genotipo": genotipo,
            "estagio_preprocessamento": estagio,
            "alvo": ALVO,
            "classe_positiva": CLASSE_POSITIVA,
            "bandas_nm": ", ".join(str(b) for b in bandas),
            "estrategia_validacao": ESTRATEGIA_VALIDACAO,
            "grupo_validacao": GRUPO_VALIDACAO,
            "turno_avaliacao": TURNO_AVALIACAO,
            "k_folds": K_FOLDS,
            "amostras": len(y),
            "semente": SEMENTE,
        }]).to_csv(saida / "configuracao.csv", sep=";", index=False)

        for arquivo, tabelas in extras_acumulados.items():
            pd.concat(tabelas, ignore_index=True).to_csv(saida / arquivo, sep=";", index=False)

    return df_resumo


def executar_modelo(
    nome: str,
    modelo_base: Pipeline,
    modo: str = MODO_PADRAO,
    estagio: str = ESTAGIO_PADRAO,
    salvar_csv: bool = True,
    extras_fn: Callable[[Pipeline, dict[str, object]], dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    print(f"Estagio de preprocessamento: {estagio}")
    print(f"Modo de bandas: {modo}")
    print(f"Turno avaliado: {TURNO_AVALIACAO}")
    print(f"Avaliacao: {ESTRATEGIA_VALIDACAO} com k={K_FOLDS} e grupo={GRUPO_VALIDACAO}")
    print(f"Salvar CSVs: {'sim' if salvar_csv else 'nao'}")

    resumos = []
    for cenario in montar_cenarios(modo, estagio):
        print("\n" + "-" * 72)
        print(f"Modelo: {nome}")
        print(f"Cenario: {cenario['modo']} | genotipo: {cenario['genotipo']}")
        print(f"Amostras: {len(cenario['y'])}")
        print(f"Bandas usadas: {', '.join(str(b) for b in cenario['bandas'])}")

        resumo = avaliar_cenario(nome, modelo_base, cenario, estagio, salvar_csv, extras_fn)
        print("\nMetricas medias nos 5 folds:")
        print(resumo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        resumos.append(resumo)

    return pd.concat(resumos, ignore_index=True)
