#!/usr/bin/env python3
"""Classificacao IRRIG vs NIRRIG dia a dia, com as bandas de cada genotipo.

Mesma montagem de `--modo por_genotipo`, quebrada por dia de coleta: para cada
genotipo, sete ajustes independentes, um por dia, sempre com as cinco bandas
que a selecao de variaveis escolheu para aquele genotipo. O que sai e a
trajetoria do estresse vista pelo classificador -- em que dia as duas condicoes
passam a ser separaveis, e o quanto.

Grupo de validacao
------------------
Aqui o grupo e o BLOCO, nao a `nomenclaura` que os outros scripts usam.
`nomenclaura` e o nome do arquivo (`B1_BR16_IRRIG_REPROD00000.asd`) e dentro de
um unico dia todos os 64 valores sao distintos -- agrupar por ele nao agruparia
nada, e as 8 leituras de uma mesma parcela cairiam dos dois lados da dobra,
inflando a acuracia. O bloco de campo e a unidade experimental real: sao 4,
entao a validacao e leave-one-block-out com k=4, o mesmo criterio que o PLS-DA
de `selecaoVariaveis` aplica e pela mesma razao.

Turno
-----
So a manha, como no resto do projeto: D02, D03 e D09 tambem tem coleta de
tarde, e mante-la daria a esses tres dias o dobro de leituras dos demais --
desbalanceamento que cairia justamente sobre o fator que a figura compara.

Uso:
    python classificacao_por_dia.py [--estagio normalizado] [--sem-csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

try:
    from .classificacao_utils import (
        ALVO, CLASSE_POSITIVA, ESTAGIO_PADRAO, GENOTIPOS, SAIDA_DIR,
        calcular_metricas, carregar, carregar_bandas_por_genotipo,
        obter_score_auc, selecionar_colunas,
    )
    from .gradient_boosting_top5 import criar_modelo as criar_gradient_boosting
    from .random_forest_top5 import criar_modelo as criar_random_forest
except ImportError:
    from classificacao_utils import (
        ALVO, CLASSE_POSITIVA, ESTAGIO_PADRAO, GENOTIPOS, SAIDA_DIR,
        calcular_metricas, carregar, carregar_bandas_por_genotipo,
        obter_score_auc, selecionar_colunas,
    )
    from gradient_boosting_top5 import criar_modelo as criar_gradient_boosting
    from random_forest_top5 import criar_modelo as criar_random_forest

TURNO = "manha"
GRUPO_VALIDACAO = "bloco"
ESTRATEGIA_VALIDACAO = "GroupKFold (leave-one-block-out)"

MODELOS = [
    ("random_forest", criar_random_forest),
    ("gradient_boosting", criar_gradient_boosting),
]


def avaliar_dia(
    nome: str,
    modelo_base: Pipeline,
    genotipo: str,
    dia: str,
    meta: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, str | None]:
    """Leave-one-block-out num unico dia. Devolve as metricas por dobra."""
    if len(np.unique(y)) != 2:
        return pd.DataFrame(), "so uma classe presente"

    grupos = meta[GRUPO_VALIDACAO].astype(str).to_numpy()
    n_blocos = len(np.unique(grupos))
    if n_blocos < 2:
        return pd.DataFrame(), f"{n_blocos} bloco(s), sem como validar"

    cv = GroupKFold(n_splits=n_blocos)
    linhas = []
    for fold, (treino, teste) in enumerate(cv.split(X, y, groups=grupos), start=1):
        # Uma dobra pode ficar com uma classe so se o bloco de fora nao tiver
        # as duas condicoes; sem duas classes nao ha AUC nem kappa.
        if len(np.unique(y[teste])) != 2 or len(np.unique(y[treino])) != 2:
            continue

        modelo = clone(modelo_base)
        modelo.fit(X[treino], y[treino])
        y_pred = modelo.predict(X[teste])
        score = obter_score_auc(modelo, X[teste])

        metrica = calcular_metricas(
            nome, "por_dia", genotipo, fold, y[teste], y_pred, score
        )
        metrica["dia"] = dia
        metrica["bloco_teste"] = str(np.unique(grupos[teste])[0])
        metrica["n_teste"] = int(len(teste))
        linhas.append(metrica)

    if not linhas:
        return pd.DataFrame(), "nenhuma dobra com as duas classes"
    return pd.DataFrame(linhas), None


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Media e desvio das dobras, por modelo, genotipo e dia."""
    metricas = ["accuracy", "precision", "recall", "f1_score", "kappa", "auc_roc"]
    agregacoes = {}
    for m in metricas:
        agregacoes[f"{m}_media"] = (m, "mean")
        agregacoes[f"{m}_desvio"] = (m, "std")
    agregacoes["dobras"] = ("fold", "size")
    agregacoes["amostras"] = ("n_teste", "sum")

    return (
        df.groupby(["modelo", "genotipo", "dia"], as_index=False)
        .agg(**agregacoes)
        .sort_values(["modelo", "genotipo", "dia"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classificacao IRRIG vs NIRRIG dia a dia, por genotipo."
    )
    parser.add_argument("--estagio", default=ESTAGIO_PADRAO,
                        choices=("recortado", "suavizado", "normalizado"))
    parser.add_argument("--sem-csv", dest="salvar_csv", action="store_false",
                        default=True, help="Executa sem criar CSVs.")
    args = parser.parse_args()

    print(f"Estagio de preprocessamento: {args.estagio}")
    print(f"Turno: {TURNO}")
    print(f"Avaliacao: {ESTRATEGIA_VALIDACAO}, grupo={GRUPO_VALIDACAO}\n")

    meta, espectro, w = carregar(args.estagio, turno=TURNO)
    print(f"{len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")

    dias = sorted(meta["dia"].unique())
    todas_metricas = []

    for genotipo in GENOTIPOS:
        bandas = carregar_bandas_por_genotipo(genotipo)
        mask_g = meta["genotipo"].astype(str).eq(genotipo).to_numpy()
        print(f"\n=== {genotipo} -- bandas {', '.join(str(b) for b in bandas)} ===")

        X_g = selecionar_colunas(espectro[mask_g], w, bandas)
        meta_g = meta.loc[mask_g].reset_index(drop=True)
        y_g = meta_g[ALVO].astype(str).to_numpy()

        for dia in dias:
            mask_d = (meta_g["dia"] == dia).to_numpy()
            meta_d = meta_g.loc[mask_d].reset_index(drop=True)
            n_pos = int((y_g[mask_d] == CLASSE_POSITIVA).sum())
            print(f"  {dia}: {int(mask_d.sum())} amostras "
                  f"({n_pos} {CLASSE_POSITIVA})", end="")

            for nome, criar in MODELOS:
                df, motivo = avaliar_dia(
                    nome, criar(), genotipo, dia, meta_d,
                    X_g[mask_d], y_g[mask_d],
                )
                if motivo:
                    print(f"  |  {nome}: pulado ({motivo})", end="")
                    continue
                todas_metricas.append(df)
                print(f"  |  {nome}: acc {df['accuracy'].mean():.3f} "
                      f"AUC {df['auc_roc'].mean():.3f}", end="")
            print()

    if not todas_metricas:
        raise SystemExit("Nenhum cenario pode ser avaliado.")

    df_folds = pd.concat(todas_metricas, ignore_index=True)
    df_resumo = resumir(df_folds)

    print("\n" + "=" * 78)
    print("Metricas por dia (media das 4 dobras):\n")
    for nome, _ in MODELOS:
        sub = df_resumo[df_resumo["modelo"] == nome]
        if sub.empty:
            continue
        print(f"--- {nome} ---")
        print(sub[["genotipo", "dia", "amostras", "accuracy_media",
                   "accuracy_desvio", "f1_score_media", "kappa_media",
                   "auc_roc_media"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    if args.salvar_csv:
        saida = SAIDA_DIR / "por_dia"
        saida.mkdir(parents=True, exist_ok=True)
        df_folds.to_csv(saida / "metricas_folds.csv", sep=";", index=False)
        df_resumo.to_csv(saida / "metricas_por_dia.csv", sep=";", index=False)

        pd.DataFrame([{
            "estagio_preprocessamento": args.estagio,
            "turno": TURNO,
            "alvo": ALVO,
            "classe_positiva": CLASSE_POSITIVA,
            "estrategia_validacao": ESTRATEGIA_VALIDACAO,
            "grupo_validacao": GRUPO_VALIDACAO,
            "modelos": ", ".join(nome for nome, _ in MODELOS),
            "bandas_por_genotipo": " | ".join(
                f"{g}: {', '.join(str(b) for b in carregar_bandas_por_genotipo(g))}"
                for g in GENOTIPOS
            ),
        }]).to_csv(saida / "configuracao.csv", sep=";", index=False)
        print(f"Resultados salvos em {saida}")


if __name__ == "__main__":
    main()
