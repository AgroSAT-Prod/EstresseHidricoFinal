#!/usr/bin/env python3
"""Redução de colinearidade entre bandas espectrais.

Bandas hiperespectrais vizinhas de 1 nm são quase cópias umas das outras: o
espectro tem 2051 colunas, mas muito menos direções independentes de variação.
Levar todas para a seleção de variáveis diluiria a importância de cada região
espectral entre dezenas de bandas redundantes.

Critério
--------
- Correlação de Spearman (monotônica, não exige linearidade nem normalidade).
- |r| > 0.80 entre todos os membros do grupo (ligação completa).
- Janela espectral de 10 nm: um grupo nunca cobre mais que 10 nm.

Os grupos são contíguos no comprimento de onda. Colinearidade em espectro é um
fenômeno local -- duas bandas distantes que aparecem correlacionadas o são por
efeito comum de amostra (albedo, ângulo de folha), não por redundância
espectral, e colapsá-las apagaria regiões inteiras do espectro. A janela de
10 nm é o que impede esse colapso.

Representante
-------------
Uma banda por grupo. Quando a saída de `testeDiferencaSignificativa` está
disponível, o representante é a banda com o menor q para o efeito de condição
(o melhor entre os dias) -- dentro de um grupo de bandas trocáveis, escolhe-se
a que mais separa irrigado de não irrigado. Sem esse arquivo, o critério cai
para o medoide: a banda mais correlacionada com as demais do grupo, ou seja,
a que melhor resume o grupo.

Saídas
------
- `grupos_colinearidade.csv`   uma linha por banda, com seu grupo
- `bandas_representativas.csv` uma linha por grupo, com o representante
- `colinearidade_resumo.csv`   totais da redução

Uso:
    python reducao_colinearidade.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"

DIFERENCAS = (
    ROOT.parent / "testeDiferencaSignificativa" / "dataset_gerado"
    / "diferencas_por_banda_dia.csv"
)

ESTAGIO_PADRAO = "normalizado"

LIMIAR_R = 0.80
JANELA_NM = 10.0


def spearman_matriz(espectro: np.ndarray) -> np.ndarray:
    """Matriz de correlação de Spearman entre todas as bandas.

    Spearman é Pearson sobre os postos, então basta postear cada coluna,
    padronizar e fazer um único produto de matrizes.
    """
    postos = np.apply_along_axis(
        lambda v: pd.Series(v).rank(method="average").to_numpy(), 0, espectro
    )
    z = postos - postos.mean(axis=0)
    norma = np.sqrt((z**2).sum(axis=0))
    # Banda constante não tem posto informativo: fica com correlação zero com
    # todo mundo e acaba isolada no próprio grupo, que é o desejado.
    norma[norma == 0] = np.inf
    z = z / norma
    return np.clip(z.T @ z, -1.0, 1.0)


def agrupar(corr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Agrupa bandas contíguas por ligação completa dentro da janela.

    Varre o espectro em ordem crescente de comprimento de onda e estende o
    grupo corrente enquanto a banda nova continuar acima do limiar contra
    *todos* os membros já aceitos e o grupo couber na janela.

    Returns:
        Vetor com o índice do grupo de cada banda.
    """
    grupo = np.zeros(len(w), dtype=int)
    membros = [0]
    atual = 0

    for j in range(1, len(w)):
        dentro_janela = w[j] - w[membros[0]] < JANELA_NM
        correlacionada = dentro_janela and np.all(
            np.abs(corr[j, membros]) > LIMIAR_R
        )
        if correlacionada:
            membros.append(j)
        else:
            atual += 1
            membros = [j]
        grupo[j] = atual

    return grupo


def carregar_prioridade(w: np.ndarray) -> pd.Series | None:
    """Menor q do efeito de condição por banda, vindo do teste por dia."""
    if not DIFERENCAS.exists():
        return None

    df = pd.read_csv(DIFERENCAS, sep=";")
    prioridade = df.groupby("banda_nm")["q_condicao"].min()
    return prioridade.reindex(w.astype(int))


def escolher_representantes(
    corr: np.ndarray,
    grupo: np.ndarray,
    prioridade: pd.Series | None,
) -> tuple[np.ndarray, str]:
    """Um representante por grupo, pelo menor q ou pelo medoide."""
    criterio = "medoide" if prioridade is None else "menor_q_condicao"
    score = (
        prioridade.to_numpy() if prioridade is not None
        else np.full(len(grupo), np.nan)
    )

    representante = np.zeros(len(grupo), dtype=bool)
    for g in np.unique(grupo):
        idx = np.flatnonzero(grupo == g)
        if len(idx) == 1:
            representante[idx[0]] = True
            continue

        s = score[idx]
        if np.isfinite(s).any():
            # Empate em q (comum quando o FDR satura) é desempatado pelo
            # medoide, que é o critério de fallback aplicado ao subconjunto.
            candidatos = idx[np.isfinite(s) & (s == np.nanmin(s))]
        else:
            candidatos = idx

        if len(candidatos) > 1:
            centralidade = np.abs(corr[np.ix_(candidatos, idx)]).mean(axis=1)
            escolhido = candidatos[int(np.argmax(centralidade))]
        else:
            escolhido = candidatos[0]
        representante[escolhido] = True

    return representante, criterio


def montar_tabelas(
    w: np.ndarray,
    corr: np.ndarray,
    grupo: np.ndarray,
    representante: np.ndarray,
    prioridade: pd.Series | None,
    criterio: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela banda a banda e tabela de grupos."""
    bandas = w.astype(int)

    r_min = np.full(len(w), np.nan)
    r_medio = np.full(len(w), np.nan)
    r_rep = np.full(len(w), np.nan)
    banda_rep = np.zeros(len(w), dtype=int)
    for g in np.unique(grupo):
        idx = np.flatnonzero(grupo == g)
        rep = idx[representante[idx]][0]
        banda_rep[idx] = bandas[rep]
        bloco = np.abs(corr[np.ix_(idx, idx)])
        if len(idx) > 1:
            fora_diagonal = bloco[~np.eye(len(idx), dtype=bool)]
            r_min[idx] = fora_diagonal.min()
            r_medio[idx] = fora_diagonal.mean()
        else:
            r_min[idx] = 1.0
            r_medio[idx] = 1.0
        r_rep[idx] = np.abs(corr[idx, rep])

    df_bandas = pd.DataFrame({
        "banda_nm": bandas,
        "grupo": grupo,
        "representante": representante,
        "banda_representante_nm": banda_rep,
        "r_abs_com_representante": r_rep,
        "r_abs_min_grupo": r_min,
        "r_abs_medio_grupo": r_medio,
        "q_condicao": (
            prioridade.to_numpy() if prioridade is not None
            else np.full(len(w), np.nan)
        ),
    })

    df_grupos = (
        df_bandas.groupby("grupo", sort=True)
        .agg(
            banda_representante_nm=("banda_representante_nm", "first"),
            tamanho=("banda_nm", "size"),
            inicio_nm=("banda_nm", "min"),
            fim_nm=("banda_nm", "max"),
            r_abs_min_grupo=("r_abs_min_grupo", "first"),
            r_abs_medio_grupo=("r_abs_medio_grupo", "first"),
        )
        .reset_index()
    )
    df_grupos["largura_nm"] = df_grupos["fim_nm"] - df_grupos["inicio_nm"]
    df_grupos["criterio_representante"] = criterio
    df_grupos["q_condicao_representante"] = (
        df_bandas.loc[df_bandas["representante"], "q_condicao"].to_numpy()
    )

    return df_bandas, df_grupos


def colinearidade_residual(corr: np.ndarray, representante: np.ndarray) -> pd.Series:
    """Quanta correlação sobra entre os representantes selecionados."""
    idx = np.flatnonzero(representante)
    bloco = np.abs(corr[np.ix_(idx, idx)])
    fora_diagonal = bloco[~np.eye(len(idx), dtype=bool)]
    return pd.Series({
        "pares": len(fora_diagonal) // 2,
        "r_abs_medio": fora_diagonal.mean(),
        "r_abs_maximo": fora_diagonal.max(),
        "pares_acima_limiar": int((fora_diagonal > LIMIAR_R).sum()) // 2,
    })


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio)
    print(f"{len(meta)} amostras x {len(w)} bandas\n")

    print("Correlação de Spearman entre todas as bandas...")
    corr = spearman_matriz(espectro)

    grupo = agrupar(corr, w)
    n_grupos = int(grupo.max()) + 1
    print(f"Agrupamento (|r| > {LIMIAR_R}, janela de {JANELA_NM:.0f} nm): "
          f"{len(w)} bandas -> {n_grupos} grupos "
          f"(redução de {1 - n_grupos / len(w):.1%})")

    prioridade = carregar_prioridade(w)
    if prioridade is None:
        print(f"AVISO: {DIFERENCAS.name} não encontrado -- representante pelo medoide.")
    representante, criterio = escolher_representantes(corr, grupo, prioridade)

    df_bandas, df_grupos = montar_tabelas(
        w, corr, grupo, representante, prioridade, criterio
    )

    tamanhos = df_grupos["tamanho"]
    print(f"Tamanho dos grupos: mediana {tamanhos.median():.0f}, "
          f"máximo {tamanhos.max()}, isolados {int((tamanhos == 1).sum())}")
    print(f"Critério do representante: {criterio}")

    residual = colinearidade_residual(corr, representante)
    print(f"\nColinearidade entre os {n_grupos} representantes: "
          f"|r| médio {residual['r_abs_medio']:.3f}, "
          f"máximo {residual['r_abs_maximo']:.3f}, "
          f"{int(residual['pares_acima_limiar'])} pares ainda acima de {LIMIAR_R}")

    resumo = pd.DataFrame([{
        "estagio": estagio,
        "amostras": len(meta),
        "bandas": len(w),
        "grupos": n_grupos,
        "reducao": 1 - n_grupos / len(w),
        "limiar_r": LIMIAR_R,
        "janela_nm": JANELA_NM,
        "criterio_representante": criterio,
        "tamanho_mediano_grupo": float(tamanhos.median()),
        "tamanho_maximo_grupo": int(tamanhos.max()),
        "grupos_isolados": int((tamanhos == 1).sum()),
        **residual.to_dict(),
    }])

    df_bandas.to_csv(SAIDA_DIR / "grupos_colinearidade.csv", sep=";", index=False)
    df_grupos.to_csv(SAIDA_DIR / "bandas_representativas.csv", sep=";", index=False)
    resumo.to_csv(SAIDA_DIR / "colinearidade_resumo.csv", sep=";", index=False)

    print(f"\nResultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
