#!/usr/bin/env python3
"""Corrige inconsistencias de metadados em Unificada13052026.csv.

Quatro inconsistencias identificadas na coluna de metadados (a leitura
espectral em si nunca e alterada):

1. `bloco` misto maiuscula/minuscula: as 192 linhas do bloco de recuperacao
   (D10M, sufixo `_RECUP` em `nomenclaura`) foram gravadas como b1-b4 em vez
   de B1-B4, criando 8 categorias para apenas 4 blocos fisicos.

2. `genotipo` errado em 32 linhas do bloco B4/D10M/RECUP: `nomenclaura`
   identifica claramente CD202 e EMB48 (`B4_CD202_..._RECUP*`,
   `B4_EMB48_..._RECUP*`), mas a coluna `genotipo` repete "BR16" para todas
   elas -- efeito de preenchimento incorreto (fill-down) no arquivo de
   origem, ja visto em versoes anteriores da base para a coluna `condicao`.

3. Typo em `nomenclaura` em 16 linhas do bloco B3/D02T: o identificador do
   genotipo foi gravado como "C202" em vez de "CD202" (a coluna `genotipo`
   nessas linhas ja estava correta).

4. `condicao` errada em 16 linhas, mesmo efeito de fill-down do caso 2:
   `B202_NIRRIG_CD2_REPROD00000-07` (D04M) e `B4_EMB48_NIRR_REP_RECUP00000-07`
   (D09T) aparecem como "IRRIG" repetindo a condicao do grupo anterior, embora
   `nomenclaura` identifique as duas series como nao irrigadas. O token de
   condicao varia entre as duas fases do experimento -- `IRRIG`/`NIRRIG` nos
   arquivos REPROD e a forma abreviada `IRR`/`NIRR` nos arquivos RECUP --,
   entao ambas as grafias sao normalizadas antes da comparacao.

Em todos os casos, `nomenclaura` (nome do arquivo .asd original) e a fonte
confiavel usada para corrigir a coluna correspondente -- exceto no caso 3,
em que o proprio `nomenclaura` e o campo com o erro de digitacao.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / "Unificada13052026.csv"
SAIDA = ROOT / "Unificada13052026_Limpa.csv"


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["bloco"] = df["bloco"].str.upper()

    tokens = df["nomenclaura"].str.split("_")

    typo_c202 = tokens.str[1] == "C202"
    tokens.loc[typo_c202] = tokens.loc[typo_c202].apply(
        lambda t: [t[0], "CD202", *t[2:]]
    )
    df.loc[typo_c202, "nomenclaura"] = tokens.loc[typo_c202].str.join("_")

    genotipo_nomenclaura = tokens.str[1]
    genotipos_validos = {"BR16", "CD202", "EMB48"}
    divergente = genotipo_nomenclaura.isin(genotipos_validos) & (
        genotipo_nomenclaura != df["genotipo"]
    )
    df.loc[divergente, "genotipo"] = genotipo_nomenclaura[divergente]

    condicoes_validas = {
        "IRRIG": "IRRIG",
        "NIRRIG": "NIRRIG",
        "IRR": "IRRIG",
        "NIRR": "NIRRIG",
    }
    condicao_nomenclaura = tokens.str[2].map(condicoes_validas)
    divergente = condicao_nomenclaura.notna() & (
        condicao_nomenclaura != df["condicao"]
    )
    df.loc[divergente, "condicao"] = condicao_nomenclaura[divergente]

    return df


def main() -> None:
    df = pd.read_csv(ENTRADA, sep=";")
    df.columns = [c.strip() for c in df.columns]

    df_limpo = limpar(df)

    n_bloco = (df["bloco"] != df_limpo["bloco"]).sum()
    n_genotipo = (df["genotipo"] != df_limpo["genotipo"]).sum()
    n_nomenclaura = (df["nomenclaura"] != df_limpo["nomenclaura"]).sum()
    n_condicao = (df["condicao"] != df_limpo["condicao"]).sum()
    print(f"bloco corrigido em {n_bloco} linhas (minusculo -> maiusculo)")
    print(f"genotipo corrigido em {n_genotipo} linhas (divergia de nomenclaura)")
    print(f"nomenclaura corrigido em {n_nomenclaura} linhas (typo C202 -> CD202)")
    print(f"condicao corrigida em {n_condicao} linhas (divergia de nomenclaura)")

    df_limpo.to_csv(SAIDA, sep=";", index=False)
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
