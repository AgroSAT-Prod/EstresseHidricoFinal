#!/usr/bin/env python3
"""Contrastes cruzados: genotipo A numa condicao contra genotipo B na outra.

`interacao_genotipo_condicao.py` cobre os efeitos simples -- os dois genotipos
do par comparados DENTRO da mesma condicao (BR16 IRRIG vs CD202 IRRIG, e o
mesmo em NIRRIG). Sobram do delineamento os seis contrastes em que genotipo e
condicao mudam ao mesmo tempo:

    BR16 IRRIG  vs CD202 NIRRIG        BR16 NIRRIG vs CD202 IRRIG
    BR16 IRRIG  vs EMB48 NIRRIG        BR16 NIRRIG vs EMB48 IRRIG
    CD202 IRRIG vs EMB48 NIRRIG        CD202 NIRRIG vs EMB48 IRRIG

Por que eles importam
---------------------
Sao os contrastes que dizem se genotipo e estresse sao confundiveis no
espectro. Se BR16 irrigado ja e espectralmente indistinguivel de CD202 sob
estresse, entao um classificador treinado sem saber o material vai ler
diferenca de genotipo como se fosse deficit hidrico. Os efeitos simples nao
respondem isso: eles medem a distancia entre materiais com a condicao fixa, e
o contraste de estresse mede a distancia entre condicoes com o material fixo.
So o cruzamento poe as duas fontes de variacao uma contra a outra.

O par de contrastes cruzados de um mesmo par de genotipos tambem nao e
redundante -- eles sao assimetricos. `A IRRIG vs B NIRRIG` soma os dois efeitos
quando eles apontam para o mesmo lado, e `A NIRRIG vs B IRRIG` os subtrai. A
diferenca entre as duas linhas e, na pratica, a mesma interacao que o ART mede,
lida na escala do contraste.

Metodo
------
Identico ao dos efeitos simples, e de proposito: mesmo teste, mesmo tamanho de
efeito, mesma familia de FDR. So o recorte das duas celulas muda, entao as
linhas desta tabela sao diretamente comparaveis com as de
`efeitos_simples.csv` -- e a figura `grade_celulas_cruzadas.png` empilha as
duas na mesma grade.

- Mann-Whitney exato banda a banda (`TESTE_EFEITO_SIMPLES`), com os postos
  recalculados dentro das duas celulas;
- FDR de Benjamini-Hochberg com as 2051 bandas como familia, uma familia por
  (dia, par, cruzamento) -- 42 no total;
- delta de Cliff como tamanho de efeito, positivo quando a celula `a` fica
  acima da celula `b`.

Vale aqui a mesma ressalva de pseudorreplicacao dos outros modulos: 32 leituras
por celula, mas 4 blocos independentes. Ordene pelo delta de Cliff, nao pelo q.

Saidas em `dataset_gerado/`:
    efeitos_cruzados.csv         - uma linha por dia, contraste e banda
    efeitos_cruzados_resumo.csv  - proporcao de bandas significativas por nivel

Uso:
    python efeitos_cruzados.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402
from diferencas_significativas import CONDICOES, GENOTIPOS  # noqa: E402
from interacao_genotipo_condicao import (  # noqa: E402
    ALPHAS,
    TESTE_EFEITO_SIMPLES,
    coluna_sig,
    contraste_celulas,
)

SAIDA_DIR = ROOT / "dataset_gerado"

ESTAGIO_PADRAO = "normalizado"

TURNO = "manha"

# Os dois cruzamentos de um par (a, b): a condicao de `a` e sempre a oposta da
# de `b`. CONDICOES vem de `diferencas_significativas` como ["IRRIG", "NIRRIG"].
CRUZAMENTOS = [(CONDICOES[0], CONDICOES[1]), (CONDICOES[1], CONDICOES[0])]

MIN_POR_CELULA = 2


def contraste_cruzado(
    Y: np.ndarray,
    genotipo: np.ndarray,
    condicao: np.ndarray,
    par: tuple[str, str],
    cruzamento: tuple[str, str],
) -> pd.DataFrame | None:
    """Genotipo `a` na condicao `cond_a` contra genotipo `b` na condicao `cond_b`.

    Recorta as duas celulas antes de chamar `contraste_celulas`, para que os
    postos sejam calculados so sobre elas -- incluir as outras quatro celulas
    do delineamento na ordenacao deslocaria o delta de Cliff.
    """
    a, b = par
    cond_a, cond_b = cruzamento

    eh_a = (genotipo == a) & (condicao == cond_a)
    eh_b = (genotipo == b) & (condicao == cond_b)
    if eh_a.sum() < MIN_POR_CELULA or eh_b.sum() < MIN_POR_CELULA:
        return None

    mask = eh_a | eh_b
    idx_a = eh_a[mask].nonzero()[0]
    idx_b = eh_b[mask].nonzero()[0]

    return contraste_celulas(Y[mask], idx_a, idx_b)


def analisar_dia(
    Y: np.ndarray,
    w: np.ndarray,
    meta_dia: pd.DataFrame,
    dia: str,
) -> pd.DataFrame:
    """Os seis contrastes cruzados de um dia, banda a banda."""
    genotipo = meta_dia["genotipo"].to_numpy()
    condicao = meta_dia["condicao"].to_numpy()
    bandas = w.astype(int)

    linhas = []
    for par in combinations(GENOTIPOS, 2):
        for cruzamento in CRUZAMENTOS:
            df = contraste_cruzado(Y, genotipo, condicao, par, cruzamento)
            if df is None:
                continue
            df.insert(0, "banda_nm", bandas)
            df.insert(0, "celula_b", f"{par[1]}|{cruzamento[1]}")
            df.insert(0, "celula_a", f"{par[0]}|{cruzamento[0]}")
            df.insert(0, "condicao_b", cruzamento[1])
            df.insert(0, "condicao_a", cruzamento[0])
            df.insert(0, "genotipo_b", par[1])
            df.insert(0, "genotipo_a", par[0])
            df.insert(0, "dia", dia)
            linhas.append(df)

    return pd.concat(linhas, ignore_index=True)


def resumir(cruzados: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dia, contraste e nivel de significancia."""
    chaves = ["dia", "genotipo_a", "condicao_a", "genotipo_b", "condicao_b"]
    partes = []

    for alpha in ALPHAS:
        agregado = cruzados.groupby(chaves).agg(
            n_bandas=("banda_nm", "size"),
            prop_sig=(coluna_sig("efeito", alpha), "mean"),
            delta_mediano=("delta_cliff", "median"),
            delta_abs_max=("delta_cliff", lambda d: d.abs().max()),
        )
        agregado.insert(0, "alpha", alpha)
        partes.append(agregado.reset_index())

    return pd.concat(partes, ignore_index=True).sort_values(
        ["genotipo_a", "genotipo_b", "condicao_a", "dia", "alpha"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def verificar_monotonicidade(resumo: pd.DataFrame) -> None:
    """Um limiar mais rigoroso nao pode declarar mais bandas significativas."""
    chaves = ["dia", "genotipo_a", "condicao_a", "genotipo_b", "condicao_b"]
    for chave, sub in resumo.groupby(chaves):
        valores = sub.sort_values("alpha", ascending=False)["prop_sig"].to_numpy()
        if np.any(np.diff(valores) > 1e-12):
            raise AssertionError(
                f"Monotonicidade violada em {chave}: {valores}"
            )


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio, turno=TURNO)
    print(f"{len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")
    print(f"Teste ativo: {TESTE_EFEITO_SIMPLES}\n")

    partes = []
    for dia in sorted(meta["dia"].unique()):
        mask = (meta["dia"] == dia).to_numpy()
        df = analisar_dia(
            espectro[mask], w, meta[mask].reset_index(drop=True), dia
        )
        partes.append(df)
        print(f"{dia} (n={int(mask.sum())}) -- "
              f"{df.groupby(['genotipo_a', 'condicao_a', 'genotipo_b']).ngroups}"
              f" contrastes cruzados x {len(w)} bandas")

    df_cruzados = pd.concat(partes, ignore_index=True)
    df_resumo = resumir(df_cruzados)
    verificar_monotonicidade(df_resumo)

    df_cruzados.to_csv(SAIDA_DIR / "efeitos_cruzados.csv", sep=";", index=False)
    df_resumo.to_csv(
        SAIDA_DIR / "efeitos_cruzados_resumo.csv", sep=";", index=False
    )

    print("\n" + "=" * 96)
    print("Bandas significativas por contraste cruzado (%)")
    print("=" * 96)
    for (a, b), sub in df_resumo.groupby(["genotipo_a", "genotipo_b"], sort=True):
        print(f"\n{a} vs {b}")
        cabecalho = "".join(f"{f'q<{al:g}':>10}" for al in ALPHAS)
        print(f"  {'dia':<5}{'contraste':<30}{cabecalho}{'delta med':>12}")
        for dia in sorted(sub["dia"].unique()):
            for cond_a, cond_b in CRUZAMENTOS:
                linhas = sub[(sub["dia"] == dia)
                             & (sub["condicao_a"] == cond_a)
                             & (sub["condicao_b"] == cond_b)]
                if linhas.empty:
                    continue
                rotulo = f"{a} {cond_a} vs {b} {cond_b}"
                props = "".join(
                    f"{linhas[linhas['alpha'] == al]['prop_sig'].iloc[0]:>9.0%} "
                    for al in ALPHAS
                )
                delta = linhas["delta_mediano"].iloc[0]
                print(f"  {dia:<5}{rotulo:<30}{props}{delta:>11.2f}")

    print("\n" + "=" * 96)
    print("Assimetria entre os dois cruzamentos de cada par")
    print("=" * 96)
    print("  Se os dois cruzamentos dessem a mesma coisa, genotipo e condicao")
    print("  seriam aditivos no espectro. A diferenca e a interacao, lida na")
    print("  escala do contraste.\n")
    referencia = df_resumo[df_resumo["alpha"] == ALPHAS[0]]
    for (a, b), sub in referencia.groupby(["genotipo_a", "genotipo_b"], sort=True):
        soma = sub[sub["condicao_a"] == CRUZAMENTOS[0][0]].set_index("dia")
        subtrai = sub[sub["condicao_a"] == CRUZAMENTOS[1][0]].set_index("dia")
        dias = sorted(set(soma.index) & set(subtrai.index))
        diffs = [abs(soma.loc[d, "prop_sig"] - subtrai.loc[d, "prop_sig"])
                 for d in dias]
        print(f"  {a} vs {b:<8} diferenca em bandas significativas: "
              f"mediana {np.median(diffs):.0%}, maxima {max(diffs):.0%} "
              f"(dia {dias[int(np.argmax(diffs))]})")

    print(f"\nResultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
