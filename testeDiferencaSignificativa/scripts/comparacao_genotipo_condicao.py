#!/usr/bin/env python3
"""Comparação das seis células genótipo x condição, dia a dia, por Kruskal-Wallis.

O teste de normalidade fechou a questão para este dataset: nenhuma banda é
normal em todos os estratos, e em `diferencas_resumo.csv` a coluna `via_anova`
é zero nos sete dias -- os 14.357 testes banda x dia foram todos pela via de
posto. Aqui a via não-paramétrica não é alternativa, é a única.

A pergunta é, para cada dia (D02..D10) e cada banda, se as seis células do
delineamento (3 genótipos x 2 condições) diferem entre si:

- dentro do genótipo: BR16 IRRIG vs BR16 NIRRIG, e o mesmo para CD202 e EMB48
  -- o efeito do estresse hídrico em cada material;
- entre genótipos: cada célula contra todas as outras -- se os materiais se
  separam espectralmente, e se essa separação depende da condição.

Análise
-------
1. Omnibus: Kruskal-Wallis sobre as 6 células combinadas.
2. Contraste de estresse: Kruskal-Wallis de duas amostras (IRRIG vs NIRRIG)
   *dentro* de cada genótipo, rodado isolado e com família de FDR própria. É o
   contraste que a pergunta nomeia primeiro, e rodá-lo fora do pós-hoc evita
   que ele pague o preço das 15 comparações do Dunn.
3. Pós-hoc de Dunn nos 15 pares das 6 células, nas bandas cujo omnibus
   sobreviveu ao FDR. Cada par é classificado em `estresse` (3 pares),
   `genotipo` (6, mesma condição) ou `cruzado` (6, genótipo e condição
   variando juntos).

Todos os p-valores recebem correção FDR de Benjamini-Hochberg, com as bandas
como família de testes.

Tamanho de efeito
-----------------
Com o n deste experimento quase toda banda dá significativa, então o q-valor
não ordena nada. Cada teste sai acompanhado de sua magnitude: epsilon^2 para o
Kruskal-Wallis, delta de Cliff para os contrastes de duas amostras (o sinal dá
a direção) e r = |z|/sqrt(N) para os pares de Dunn. É por essas colunas que as
bandas devem ser ordenadas, não pelo p.

Pseudorreplicação
-----------------
Cada célula genótipo x condição tem 32 a 64 leituras, mas apenas 4 blocos
independentes: as 8 leituras de um mesmo bloco são subamostras da mesma
parcela. O n efetivo está inflado de 8 a 16 vezes, e é isso que produz a taxa
de ~97% de bandas significativas do módulo atual. A leitura é mantida como
unidade porque é o que o delineamento deste módulo já usa, mas os tamanhos de
efeito acima existem justamente para que a leitura dos resultados não dependa
de p-valores inflados. Ver `UNIDADE_BLOCO` para trocar a unidade.

Uso:
    python comparacao_genotipo_condicao.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402
from selecao_estatistica import fdr_bh  # noqa: E402
from diferencas_significativas import (  # noqa: E402
    ALPHA,
    CONDICOES,
    GENOTIPOS,
    correcao_empates,
    dunn_vetorizado,
    kruskal_vetorizado,
)

SAIDA_DIR = ROOT.parent / "resultados" / "dataset_gerado"

ESTAGIO_PADRAO = "normalizado"

# Único turno presente em todos os sete dias. Manter manhã e tarde juntas
# deixaria D02, D03 e D09 com o dobro de leituras dos demais dias, e o
# desbalanceamento cairia sobre a comparação entre dias.
TURNO = "manha"

# Trocar para True agrega as 8 leituras de cada bloco x genótipo x condição na
# média antes de testar, o que elimina a pseudorreplicação ao custo de reduzir
# cada célula a 4 unidades.
UNIDADE_BLOCO = False

REGIOES = [
    ("VIS", 400, 700),
    ("red edge", 700, 780),
    ("NIR", 780, 1350),
    ("SWIR1", 1350, 1800),
    ("SWIR2", 1800, 2451),
]


def rotulo_celula(genotipo: str, condicao: str) -> str:
    """Nome da célula do delineamento, no mesmo formato do pós-hoc existente."""
    return f"{genotipo}|{condicao}"


def tipo_par(a: str, b: str) -> str:
    """Classifica um par de células pela pergunta que ele responde."""
    genotipo_a, condicao_a = a.split("|")
    genotipo_b, condicao_b = b.split("|")
    if genotipo_a == genotipo_b:
        return "estresse"
    if condicao_a == condicao_b:
        return "genotipo"
    return "cruzado"


def epsilon_quadrado(H: np.ndarray, n: int, k: int) -> np.ndarray:
    """Tamanho de efeito do Kruskal-Wallis, na escala [0, 1].

    epsilon^2 = (H - k + 1) / (n - k): a fração da variabilidade dos postos
    atribuível ao agrupamento, corrigida pelos graus de liberdade.
    """
    return np.clip((H - k + 1) / (n - k), 0.0, 1.0)


def cliff_delta(
    postos: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    """Delta de Cliff de a contra b, a partir das somas de postos.

    delta = 2U/(na*nb) - 1, com U de Mann-Whitney obtido da soma de postos de
    a. Vale de -1 a +1 e é a probabilidade de um valor de a superar um de b
    menos a probabilidade do contrário -- positivo significa que a tende a
    ficar acima de b. Exige que os postos tenham sido calculados sobre a união
    de a e b, e só de a e b.
    """
    n_a, n_b = len(idx_a), len(idx_b)
    soma_a = postos[idx_a].sum(axis=0)
    u_a = soma_a - n_a * (n_a + 1) / 2.0
    return 2.0 * u_a / (n_a * n_b) - 1.0


def agregar_por_bloco(
    espectro: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Média das leituras de cada bloco x genótipo x condição x dia."""
    chaves = ["bloco", "genotipo", "condicao", "dia"]
    grupos = meta.groupby(chaves, sort=True).indices

    linhas = []
    medias = np.empty((len(grupos), espectro.shape[1]))
    for i, (valores, idx) in enumerate(sorted(grupos.items())):
        medias[i] = espectro[idx].mean(axis=0)
        linhas.append(dict(zip(chaves, valores)))

    return medias, pd.DataFrame(linhas)


def celulas_do_dia(meta_dia: pd.DataFrame) -> dict[str, np.ndarray]:
    """Índices das seis células genótipo x condição presentes no dia."""
    genotipo = meta_dia["genotipo"].to_numpy()
    condicao = meta_dia["condicao"].to_numpy()

    celulas = {
        rotulo_celula(g, c): ((genotipo == g) & (condicao == c)).nonzero()[0]
        for g in GENOTIPOS
        for c in CONDICOES
    }
    return {nome: idx for nome, idx in celulas.items() if len(idx) > 0}


def omnibus_do_dia(
    Y: np.ndarray,
    w: np.ndarray,
    dia: str,
    celulas: dict[str, np.ndarray],
    postos: np.ndarray,
    empates: np.ndarray,
) -> pd.DataFrame:
    """Kruskal-Wallis sobre as 6 células combinadas, banda a banda."""
    H, p = kruskal_vetorizado(postos, list(celulas.values()), empates)
    q = np.clip(fdr_bh(p), 0, 1)

    return pd.DataFrame({
        "dia": dia,
        "banda_nm": w.astype(int),
        "n": len(Y),
        "celulas": len(celulas),
        "H": H,
        "p_valor": p,
        "q_fdr": q,
        "epsilon2": epsilon_quadrado(H, len(Y), len(celulas)),
        "significativa": q < ALPHA,
    })


def estresse_do_dia(
    Y: np.ndarray,
    w: np.ndarray,
    dia: str,
    meta_dia: pd.DataFrame,
) -> pd.DataFrame:
    """IRRIG vs NIRRIG dentro de cada genótipo, com FDR próprio por genótipo.

    Os postos são recalculados dentro do genótipo: o contraste é entre as duas
    condições daquele material, e incluir os outros genótipos na ordenação
    contaminaria tanto o teste quanto o delta de Cliff.
    """
    genotipo = meta_dia["genotipo"].to_numpy()
    condicao = meta_dia["condicao"].to_numpy()

    partes = []
    for g in GENOTIPOS:
        mask = genotipo == g
        if not mask.any():
            continue

        Y_gen = Y[mask]
        cond_gen = condicao[mask]
        idx_irrig = (cond_gen == "IRRIG").nonzero()[0]
        idx_nirrig = (cond_gen == "NIRRIG").nonzero()[0]
        if len(idx_irrig) < 2 or len(idx_nirrig) < 2:
            continue

        postos = rankdata(Y_gen, axis=0)
        empates = correcao_empates(Y_gen)

        H, p = kruskal_vetorizado(postos, [idx_irrig, idx_nirrig], empates)
        q = np.clip(fdr_bh(p), 0, 1)

        partes.append(pd.DataFrame({
            "dia": dia,
            "genotipo": g,
            "banda_nm": w.astype(int),
            "n_irrig": len(idx_irrig),
            "n_nirrig": len(idx_nirrig),
            "H": H,
            "p_valor": p,
            "q_fdr": q,
            "epsilon2": epsilon_quadrado(H, len(Y_gen), 2),
            # Positivo: NIRRIG com valores acima de IRRIG naquela banda.
            "delta_cliff": cliff_delta(postos, idx_nirrig, idx_irrig),
            "significativa": q < ALPHA,
        }))

    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def pares_do_dia(
    Y: np.ndarray,
    w: np.ndarray,
    dia: str,
    celulas: dict[str, np.ndarray],
    postos: np.ndarray,
    empates: np.ndarray,
    significativa: np.ndarray,
) -> pd.DataFrame:
    """Pós-hoc de Dunn nos 15 pares, restrito às bandas com omnibus significativo."""
    if not significativa.any():
        return pd.DataFrame()

    dunn = dunn_vetorizado(
        postos[:, significativa], celulas, empates[significativa]
    )
    bandas = w[significativa].astype(int)
    n = len(Y)

    linhas = []
    for (a, b), (z, p) in dunn.items():
        linhas.append(pd.DataFrame({
            "dia": dia,
            "banda_nm": bandas,
            "celula_a": a,
            "celula_b": b,
            "tipo": tipo_par(a, b),
            "z": z,
            "p_valor": p,
            # r de Rosenthal: o z de Dunn na escala de correlação.
            "r_efeito": np.abs(z) / np.sqrt(n),
        }))

    df = pd.concat(linhas, ignore_index=True)
    # A família do pós-hoc são todas as comparações do dia.
    df["q_fdr"] = np.clip(fdr_bh(df["p_valor"].to_numpy()), 0, 1)
    df["significativa"] = df["q_fdr"] < ALPHA
    return df


def resumir_pares(df_pares: pd.DataFrame, bandas_por_dia: dict[str, int]) -> pd.DataFrame:
    """Uma linha por dia e par, com quantas bandas separam as duas células."""
    linhas = []
    for (dia, a, b), sub in df_pares.groupby(
        ["dia", "celula_a", "celula_b"], sort=True
    ):
        n_sig = int(sub["significativa"].sum())
        total = bandas_por_dia[dia]
        linhas.append({
            "dia": dia,
            "celula_a": a,
            "celula_b": b,
            "tipo": sub["tipo"].iloc[0],
            "bandas_testadas": len(sub),
            "bandas_totais": total,
            "bandas_sig": n_sig,
            "prop_sig": n_sig / total,
            "r_mediano": float(sub["r_efeito"].median()),
            "r_maximo": float(sub["r_efeito"].max()),
            "banda_r_maximo": int(sub.loc[sub["r_efeito"].idxmax(), "banda_nm"]),
        })
    return pd.DataFrame(linhas).sort_values(
        ["dia", "prop_sig"], ascending=[True, False]
    )


def regiao_da_banda(bandas: np.ndarray) -> np.ndarray:
    """Rótulo da região espectral de cada banda."""
    regiao = np.full(len(bandas), "fora", dtype=object)
    for nome, ini, fim in REGIOES:
        regiao[(bandas >= ini) & (bandas < fim)] = nome
    return regiao


def resumir_regioes(df_pares: pd.DataFrame) -> pd.DataFrame:
    """Onde no espectro cada par se separa."""
    df = df_pares.copy()
    df["regiao"] = regiao_da_banda(df["banda_nm"].to_numpy())

    resumo = (
        df.groupby(["dia", "celula_a", "celula_b", "tipo", "regiao"], sort=True)
        .agg(
            bandas=("significativa", "size"),
            bandas_sig=("significativa", "sum"),
            r_mediano=("r_efeito", "median"),
        )
        .reset_index()
    )
    resumo["prop_sig"] = resumo["bandas_sig"] / resumo["bandas"]
    return resumo


def imprimir_dia(
    dia: str,
    df_omnibus: pd.DataFrame,
    df_estresse: pd.DataFrame,
    df_pares: pd.DataFrame,
) -> None:
    """Relatório de console de um dia."""
    n_bandas = len(df_omnibus)
    print(f"\n{dia} (n={int(df_omnibus['n'].iloc[0])})")
    print(f"  omnibus das 6 celulas: {int(df_omnibus['significativa'].sum())} "
          f"de {n_bandas} bandas (q<{ALPHA}), "
          f"epsilon2 mediano {df_omnibus['epsilon2'].median():.3f}")

    if not df_estresse.empty:
        print("  IRRIG vs NIRRIG dentro do genotipo:")
        for g, sub in df_estresse.groupby("genotipo", sort=True):
            n_sig = int(sub["significativa"].sum())
            print(f"    {g:<6} {n_sig:4d}/{len(sub)} bandas ({n_sig / len(sub):5.1%})  "
                  f"epsilon2 mediano {sub['epsilon2'].median():.3f}  "
                  f"delta de Cliff mediano {sub['delta_cliff'].median():+.3f}")

    if df_pares.empty:
        return

    print("  pares de Dunn (ordenados por bandas significativas):")
    resumo = (
        df_pares.groupby(["celula_a", "celula_b", "tipo"], sort=False)
        .agg(sig=("significativa", "sum"), total=("significativa", "size"),
             r=("r_efeito", "median"))
        .reset_index()
        .sort_values("sig", ascending=False)
    )
    for _, row in resumo.iterrows():
        marca = " <<" if row["tipo"] == "estresse" else ""
        print(f"    {row['celula_a']:<13} vs {row['celula_b']:<13} "
              f"[{row['tipo']:<9}] {int(row['sig']):4d}/{int(row['total'])} "
              f"({row['sig'] / row['total']:5.1%})  r mediano {row['r']:.3f}{marca}")


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio, turno=TURNO)
    if TURNO is not None:
        print(f"Restrito ao turno '{TURNO}'")

    if UNIDADE_BLOCO:
        espectro, meta = agregar_por_bloco(espectro, meta)
        print("Unidade de análise: média por bloco x genotipo x condicao")
    else:
        print("Unidade de análise: leitura individual")

    print(f"{len(meta)} amostras x {len(w)} bandas")

    omnibus, estresse, pares = [], [], []
    bandas_por_dia: dict[str, int] = {}

    for dia in sorted(meta["dia"].unique()):
        mask = (meta["dia"] == dia).to_numpy()
        meta_dia = meta[mask].reset_index(drop=True)
        Y = espectro[mask]

        celulas = celulas_do_dia(meta_dia)
        postos = rankdata(Y, axis=0)
        empates = correcao_empates(Y)

        df_omnibus = omnibus_do_dia(Y, w, dia, celulas, postos, empates)
        df_estresse = estresse_do_dia(Y, w, dia, meta_dia)
        df_pares = pares_do_dia(
            Y, w, dia, celulas, postos, empates,
            df_omnibus["significativa"].to_numpy(),
        )

        bandas_por_dia[dia] = len(df_omnibus)
        omnibus.append(df_omnibus)
        if not df_estresse.empty:
            estresse.append(df_estresse)
        if not df_pares.empty:
            pares.append(df_pares)

        imprimir_dia(dia, df_omnibus, df_estresse, df_pares)

    df_omnibus = pd.concat(omnibus, ignore_index=True)
    df_estresse = pd.concat(estresse, ignore_index=True)
    df_pares = pd.concat(pares, ignore_index=True)
    df_resumo = resumir_pares(df_pares, bandas_por_dia)
    df_regioes = resumir_regioes(df_pares)

    df_omnibus.to_csv(SAIDA_DIR / "comparacao_omnibus.csv", sep=";", index=False)
    df_estresse.to_csv(SAIDA_DIR / "comparacao_estresse.csv", sep=";", index=False)
    df_pares.to_csv(SAIDA_DIR / "comparacao_pares.csv", sep=";", index=False)
    df_resumo.to_csv(SAIDA_DIR / "comparacao_resumo.csv", sep=";", index=False)
    df_regioes.to_csv(SAIDA_DIR / "comparacao_regioes.csv", sep=";", index=False)

    print("\nEfeito do estresse por genotipo ao longo dos dias "
          "(% de bandas com IRRIG != NIRRIG):")
    tabela = (
        df_estresse.groupby(["genotipo", "dia"])["significativa"].mean()
        .unstack("dia") * 100
    )
    print(tabela.round(1).to_string())

    print("\nMaior separacao por tipo de par:")
    for tipo, sub in df_resumo.groupby("tipo", sort=True):
        topo = sub.nlargest(1, "prop_sig").iloc[0]
        print(f"  {tipo:<9} {topo['dia']}  {topo['celula_a']} vs "
              f"{topo['celula_b']}  {topo['prop_sig']:.1%} das bandas")

    print(f"\nResultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
