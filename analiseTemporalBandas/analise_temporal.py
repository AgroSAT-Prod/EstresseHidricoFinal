#!/usr/bin/env python3
"""Análise temporal das bandas espectrais (blocospot temporal).

Avalia a dinâmica espectral ao longo do experimento tratando o dia de coleta
como fator principal e a unidade experimental como bloco temporal (medidas
repetidas).

Delineamento
------------
O fator principal é `dia` (D02..D10 -- o experimento não tem D0/D01; D02 é a
primeira coleta e faz o papel de linha de base). O bloco temporal é a unidade
experimental `bloco|genotipo|condicao` (24 unidades), medida em todos os 7
dias, o que dá um painel completo e balanceado de 24 sujeitos x 7 dias.

Duas decisões de recorte tornam o painel comparável entre dias:

- Apenas o turno da manhã entra na análise. Só 3 dos 7 dias têm coleta de
  tarde; misturar turnos faria a variação diurna se confundir com a evolução
  temporal, e o desbalanceamento cairia justamente sobre o fator de interesse.
- As 8 leituras de cada unidade em cada dia são reduzidas à média. A leitura
  individual é subamostra, não repetição do delineamento: usá-la como unidade
  inflaria os graus de liberdade do resíduo (pseudorreplicação).

Análise
-------
- Dados normais (Shapiro-Wilk não rejeita H0 nos resíduos do modelo aditivo):
  ANOVA de medidas repetidas de um fator (dia), com a variação entre unidades
  removida como bloco. A esfericidade é corrigida por Greenhouse-Geisser, e o
  p-valor GG é o que decide a significância -- com 7 níveis de tempo e apenas
  24 unidades, assumir esfericidade seria antiliberal demais.

- Dados não normais: Friedman sobre os postos dentro de cada unidade, que é o
  análogo de posto da ANOVA de medidas repetidas.

Pós-hoc
-------
Todas as 21 comparações par a par entre os 7 dias, incluindo o subconjunto
D02 vs Dn (linha de base contra cada dia seguinte), sinalizado na coluna
`baseline`. Tukey HSD (distribuição da amplitude studentizada com o QM do
resíduo intra-unidades) na via normal; Nemenyi, o equivalente de posto do
Tukey, na via de Friedman.

A análise roda três vezes: com todas as unidades, e dentro de cada condição
(IRRIG, NIRRIG) separadamente -- é a comparação entre esses dois recortes que
separa a evolução temporal do estresse hídrico da evolução fenológica comum
aos dois tratamentos.

Resultados
----------
- `temporal_por_banda.csv`     estatística de tempo banda a banda
- `tukey_posthoc.csv`          as 21 comparações par a par por banda
- `separacao_temporal.csv`     separação espectral de cada par de dias
- `bandas_sensiveis.csv`       bandas ordenadas por sensibilidade temporal
- `temporal_resumo.csv`        uma linha por grupo analisado

Uso:
    python analise_temporal.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, rankdata, studentized_range

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT.parent / "testeDiferencaSignificativa"))

from shapiro_normalidade import carregar, shapiro_por_banda  # noqa: E402
from selecao_estatistica import fdr_bh  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"

ESTAGIO_PADRAO = "normalizado"

ALPHA = 0.05

# Único turno presente em todos os sete dias.
TURNO = "manha"

# Chaves da unidade experimental que funciona como bloco temporal.
UNIDADE = ["bloco", "genotipo", "condicao"]

# Quantas bandas mais sensíveis ao tempo são reportadas por grupo.
TOP_BANDAS = 100


def montar_painel(
    meta: pd.DataFrame,
    espectro: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Reduz as leituras a um painel (unidade, dia, banda) de médias.

    Returns:
        Y de forma (n_unidades, n_dias, n_bandas), a lista de unidades e a
        lista de dias, ambas na ordem dos eixos de Y.
    """
    chaves = meta[UNIDADE].astype(str).agg("|".join, axis=1)
    unidades = sorted(chaves.unique())
    dias = sorted(meta["dia"].unique())

    Y = np.full((len(unidades), len(dias), espectro.shape[1]), np.nan)
    for i, unidade in enumerate(unidades):
        for j, dia in enumerate(dias):
            mask = ((chaves == unidade) & (meta["dia"] == dia)).to_numpy()
            if mask.any():
                Y[i, j] = espectro[mask].mean(axis=0)

    return Y, unidades, dias


def anova_medidas_repetidas(Y: np.ndarray) -> dict[str, np.ndarray]:
    """ANOVA de um fator com medidas repetidas, vetorizada nas bandas.

    Y tem forma (n_unidades, n_dias, n_bandas). A variação entre unidades sai
    do resíduo como bloco, que é o ganho de precisão das medidas repetidas
    sobre uma ANOVA de um fator comum.
    """
    n, k, _ = Y.shape

    geral = Y.mean(axis=(0, 1))
    media_dia = Y.mean(axis=0)
    media_unidade = Y.mean(axis=1)

    sq_total = ((Y - geral) ** 2).sum(axis=(0, 1))
    sq_dia = n * ((media_dia - geral) ** 2).sum(axis=0)
    sq_unidade = k * ((media_unidade - geral) ** 2).sum(axis=0)
    sq_residuo = sq_total - sq_dia - sq_unidade

    gl_dia = k - 1
    gl_residuo = (n - 1) * (k - 1)
    qm_residuo = sq_residuo / gl_residuo

    with np.errstate(divide="ignore", invalid="ignore"):
        F = (sq_dia / gl_dia) / qm_residuo
        eta2 = sq_dia / (sq_dia + sq_residuo)

    epsilon = greenhouse_geisser(Y)
    p = f_dist.sf(F, gl_dia, gl_residuo)
    p_gg = f_dist.sf(F, gl_dia * epsilon, gl_residuo * epsilon)

    finito = np.isfinite(F)
    return {
        "F": F,
        "p": np.where(finito, p, np.nan),
        "epsilon_gg": epsilon,
        "p_gg": np.where(finito, p_gg, np.nan),
        "eta2_parcial": eta2,
        "qm_residuo": qm_residuo,
        "gl_residuo": np.full(Y.shape[2], gl_residuo),
        "media_dia": media_dia,
    }


def greenhouse_geisser(Y: np.ndarray) -> np.ndarray:
    """Epsilon de Greenhouse-Geisser por banda.

    Mede o quanto a matriz de covariância entre os dias se afasta da
    esfericidade: epsilon = 1 sob esfericidade perfeita e cai até o limite
    inferior 1/(k-1) no pior caso. Multiplica os graus de liberdade do teste F.
    """
    n, k, p = Y.shape

    centrado = Y - Y.mean(axis=0, keepdims=True)
    cov = np.einsum("nip,njp->ijp", centrado, centrado) / (n - 1)

    # Dupla centragem da matriz de covariância (linhas, colunas e média geral).
    cov = (
        cov
        - cov.mean(axis=0, keepdims=True)
        - cov.mean(axis=1, keepdims=True)
        + cov.mean(axis=(0, 1), keepdims=True)
    )

    traco = np.trace(cov, axis1=0, axis2=1)
    soma_quadrados = (cov**2).sum(axis=(0, 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = traco**2 / ((k - 1) * soma_quadrados)

    return np.clip(np.nan_to_num(epsilon, nan=1.0), 1.0 / (k - 1), 1.0)


def residuos_aditivos(Y: np.ndarray) -> np.ndarray:
    """Resíduos do modelo dia + unidade, achatados em (n*k, n_bandas)."""
    residuo = (
        Y
        - Y.mean(axis=0, keepdims=True)
        - Y.mean(axis=1, keepdims=True)
        + Y.mean(axis=(0, 1), keepdims=True)
    )
    return residuo.reshape(-1, Y.shape[2])


def correcao_empates_postos(postos: np.ndarray) -> np.ndarray:
    """Soma de (t^3 - t) sobre os empates dentro de cada unidade, por banda."""
    n, k, p = postos.shape
    soma = np.zeros(p)
    ordenado = np.sort(postos, axis=1)
    if not np.any(ordenado[:, 1:] == ordenado[:, :-1]):
        return soma

    for j in range(p):
        for i in range(n):
            _, contagens = np.unique(postos[i, :, j], return_counts=True)
            t = contagens[contagens > 1].astype(float)
            soma[j] += np.sum(t**3 - t)
    return soma


def friedman(Y: np.ndarray) -> dict[str, np.ndarray]:
    """Teste de Friedman vetorizado nas bandas, com correção de empates.

    Os postos são atribuídos dentro de cada unidade experimental, o que é
    exatamente o bloqueio temporal em escala de posto.
    """
    n, k, _ = Y.shape

    postos = rankdata(Y, axis=1)
    soma_postos = postos.sum(axis=0)
    media_postos = postos.mean(axis=0)

    estat = (
        12.0 / (n * k * (k + 1)) * (soma_postos**2).sum(axis=0)
        - 3.0 * n * (k + 1)
    )

    empates = correcao_empates_postos(postos)
    correcao = 1.0 - empates / (n * k * (k**2 - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        estat = np.where(correcao > 0, estat / correcao, np.nan)

    # W de Kendall: a estatística de Friedman reescalada para [0, 1], que é a
    # medida de tamanho de efeito comparável ao eta^2 da via paramétrica.
    w_kendall = estat / (n * (k - 1))

    return {
        "estat": estat,
        "p": chi2.sf(estat, k - 1),
        "w_kendall": np.clip(w_kendall, 0, 1),
        "media_postos": media_postos,
    }


def p_amplitude_studentizada(
    q: np.ndarray,
    k: int,
    gl: float,
    onde: np.ndarray,
) -> np.ndarray:
    """sf da amplitude studentizada, avaliada só nas bandas de `onde`.

    A sf não tem forma fechada -- cada avaliação é uma quadratura numérica --
    e cada via só precisa dos p-valores das bandas que ela governa, então
    restringir a máscara corta o custo do pós-hoc pela metade.
    """
    p = np.full(len(q), np.nan)
    mask = onde & np.isfinite(q)
    if mask.any():
        p[mask] = studentized_range.sf(q[mask], k, gl)
    return p


def tukey_hsd(
    media_dia: np.ndarray,
    qm_residuo: np.ndarray,
    n: int,
    k: int,
    gl_residuo: int,
    onde: np.ndarray,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Tukey HSD em todos os pares de dias, vetorizado nas bandas.

    Returns:
        Mapa (i, j) -> (diferença de médias, q studentizado, p-valor).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        erro_padrao = np.sqrt(qm_residuo / n)

    resultado = {}
    for i in range(k):
        for j in range(i + 1, k):
            diferenca = media_dia[i] - media_dia[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.abs(diferenca) / erro_padrao
            resultado[(i, j)] = (
                diferenca, q, p_amplitude_studentizada(q, k, gl_residuo, onde)
            )
    return resultado


def nemenyi(
    media_postos: np.ndarray,
    n: int,
    k: int,
    onde: np.ndarray,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pós-hoc de Nemenyi -- o Tukey HSD sobre os postos de Friedman.

    Usa a mesma distribuição da amplitude studentizada, com graus de liberdade
    infinitos, porque o erro padrão dos postos é conhecido pelo delineamento e
    não estimado dos dados.
    """
    erro_padrao = np.sqrt(k * (k + 1) / (6.0 * n))

    resultado = {}
    for i in range(k):
        for j in range(i + 1, k):
            diferenca = media_postos[i] - media_postos[j]
            q = np.abs(diferenca) / erro_padrao
            resultado[(i, j)] = (
                diferenca, q, p_amplitude_studentizada(q, k, np.inf, onde)
            )
    return resultado


def analisar_grupo(
    Y: np.ndarray,
    w: np.ndarray,
    dias: list[str],
    grupo: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda a via paramétrica ou de posto banda a banda para um grupo."""
    n, k, n_bandas = Y.shape

    residuos = residuos_aditivos(Y)
    _, p_shapiro = shapiro_por_banda(residuos)
    normal = np.isfinite(p_shapiro) & (p_shapiro > ALPHA)
    via = np.where(normal, "anova_mr", "friedman")

    rm = anova_medidas_repetidas(Y)
    fr = friedman(Y)

    # A estatística de cada via só é reportada nas bandas que a via governa,
    # mas as duas são calculadas sobre todas as bandas porque o custo é o
    # mesmo e o pós-hoc precisa do QM do resíduo de qualquer forma.
    p_tempo = np.where(normal, rm["p_gg"], fr["p"])
    efeito = np.where(normal, rm["eta2_parcial"], fr["w_kendall"])

    df = pd.DataFrame({
        "grupo": grupo,
        "banda_nm": w.astype(int),
        "n_unidades": n,
        "n_dias": k,
        "via": via,
        "p_shapiro_residuo": p_shapiro,
        "F_anova_mr": np.where(normal, rm["F"], np.nan),
        "epsilon_gg": np.where(normal, rm["epsilon_gg"], np.nan),
        "p_anova_mr": np.where(normal, rm["p"], np.nan),
        "p_anova_mr_gg": np.where(normal, rm["p_gg"], np.nan),
        "eta2_parcial": np.where(normal, rm["eta2_parcial"], np.nan),
        "chi2_friedman": np.where(normal, np.nan, fr["estat"]),
        "p_friedman": np.where(normal, np.nan, fr["p"]),
        "w_kendall": np.where(normal, np.nan, fr["w_kendall"]),
        "p_tempo": p_tempo,
        "efeito_tempo": efeito,
    })

    # FDR dentro de cada via: as bandas são a família de testes, e misturar as
    # vias faria a correção depender da decisão de normalidade.
    q = np.full(n_bandas, np.nan)
    for rotulo in ("anova_mr", "friedman"):
        mask = via == rotulo
        if mask.any():
            q[mask] = np.clip(fdr_bh(p_tempo[mask]), 0, 1)
    df["q_tempo"] = q
    df["significativa"] = df["q_tempo"] < ALPHA

    df_posthoc = posthoc_grupo(Y, w, dias, grupo, normal, rm, fr)
    return df, df_posthoc


def posthoc_grupo(
    Y: np.ndarray,
    w: np.ndarray,
    dias: list[str],
    grupo: str,
    normal: np.ndarray,
    rm: dict[str, np.ndarray],
    fr: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Monta as 21 comparações par a par de dias para todas as bandas."""
    n, k, _ = Y.shape
    gl_residuo = int(rm["gl_residuo"][0])

    tukey = tukey_hsd(
        rm["media_dia"], rm["qm_residuo"], n, k, gl_residuo, onde=normal
    )
    nem = nemenyi(fr["media_postos"], n, k, onde=~normal)

    bandas = w.astype(int)
    linhas = []
    for (i, j) in tukey:
        dif_t, q_t, p_t = tukey[(i, j)]
        dif_n, q_n, p_n = nem[(i, j)]
        linhas.append(pd.DataFrame({
            "grupo": grupo,
            "banda_nm": bandas,
            "dia_a": dias[i],
            "dia_b": dias[j],
            # A primeira coleta é a linha de base do experimento: estas são as
            # comparações D02 vs Dn pedidas na metodologia.
            "baseline": i == 0,
            "via": np.where(normal, "tukey", "nemenyi"),
            # A diferença de médias fica na escala do espectro e é reportada
            # para todas as bandas: é dela que sai a distância espectral entre
            # os dias, que é descritiva e não depende da via do teste.
            "diferenca_media": dif_t,
            "diferenca_postos": dif_n,
            "estat_q": np.where(normal, q_t, q_n),
            "p_valor": np.where(normal, p_t, p_n),
        }))

    df = pd.concat(linhas, ignore_index=True)
    # A família do pós-hoc são todas as comparações do grupo (21 pares x
    # bandas), o que mantém a taxa de falsas descobertas do conjunto de
    # conclusões que se lê da tabela.
    df["q_fdr"] = np.clip(fdr_bh(df["p_valor"].to_numpy()), 0, 1)
    df["significativa"] = df["q_fdr"] < ALPHA
    return df


def separacao_temporal(df_posthoc: pd.DataFrame) -> pd.DataFrame:
    """Separação espectral de cada par de dias, para localizar o pico.

    A distância euclidiana entre os espectros médios de dois dias mede a
    separação em magnitude; a proporção de bandas significativas mede a
    extensão dela ao longo do espectro. O momento de maior separação é o par
    que maximiza as duas.
    """
    linhas = []
    for (grupo, dia_a, dia_b), sub in df_posthoc.groupby(
        ["grupo", "dia_a", "dia_b"], sort=True
    ):
        diferencas = sub["diferenca_media"].to_numpy()
        n_sig = int(sub["significativa"].sum())
        linhas.append({
            "grupo": grupo,
            "dia_a": dia_a,
            "dia_b": dia_b,
            "baseline": bool(sub["baseline"].iloc[0]),
            "bandas": len(sub),
            "bandas_sig": n_sig,
            "prop_sig": n_sig / len(sub),
            "dist_euclidiana": float(np.sqrt(np.nansum(diferencas**2)))
            if len(diferencas) else np.nan,
            "dif_abs_media": float(np.nanmean(np.abs(diferencas)))
            if len(diferencas) else np.nan,
            "estat_q_mediana": float(sub["estat_q"].median()),
        })

    df = pd.DataFrame(linhas)
    df["rank_separacao"] = (
        df.groupby("grupo")["prop_sig"].rank(ascending=False, method="min").astype(int)
    )
    return df.sort_values(["grupo", "prop_sig"], ascending=[True, False])


def bandas_sensiveis(df_banda: pd.DataFrame) -> pd.DataFrame:
    """Bandas ordenadas por sensibilidade à evolução temporal, por grupo."""
    partes = []
    for grupo, sub in df_banda.groupby("grupo", sort=True):
        sub = sub[sub["significativa"]].copy()
        sub = sub.sort_values("efeito_tempo", ascending=False).head(TOP_BANDAS)
        sub.insert(2, "posicao", np.arange(1, len(sub) + 1))
        partes.append(sub[[
            "grupo", "banda_nm", "posicao", "via", "efeito_tempo",
            "p_tempo", "q_tempo",
        ]])
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def resumir(
    df_banda: pd.DataFrame,
    df_posthoc: pd.DataFrame,
    df_sep: pd.DataFrame,
) -> pd.DataFrame:
    """Uma linha por grupo analisado."""
    linhas = []
    for grupo, sub in df_banda.groupby("grupo", sort=True):
        posthoc = df_posthoc[df_posthoc["grupo"] == grupo]
        sep = df_sep[df_sep["grupo"] == grupo]
        pico = sep.iloc[0] if len(sep) else None
        base = sep[sep["baseline"]]
        pico_base = base.iloc[0] if len(base) else None
        sensiveis = sub[sub["significativa"]].nlargest(5, "efeito_tempo")

        linhas.append({
            "grupo": grupo,
            "n_unidades": int(sub["n_unidades"].iloc[0]),
            "n_dias": int(sub["n_dias"].iloc[0]),
            "bandas": len(sub),
            "via_anova_mr": int((sub["via"] == "anova_mr").sum()),
            "via_friedman": int((sub["via"] == "friedman").sum()),
            "bandas_sig": int(sub["significativa"].sum()),
            "prop_bandas_sig": float(sub["significativa"].mean()),
            "efeito_mediano": float(sub["efeito_tempo"].median()),
            "epsilon_gg_mediano": float(sub["epsilon_gg"].median()),
            "pares_posthoc": len(posthoc),
            "pares_posthoc_sig": int(posthoc["significativa"].sum()),
            "par_maior_separacao": f"{pico['dia_a']} vs {pico['dia_b']}"
            if pico is not None else None,
            "prop_sig_maior_separacao": float(pico["prop_sig"])
            if pico is not None else None,
            "baseline_maior_separacao": f"{pico_base['dia_a']} vs {pico_base['dia_b']}"
            if pico_base is not None else None,
            "top5_bandas_nm": ", ".join(str(b) for b in sensiveis["banda_nm"]),
        })
    return pd.DataFrame(linhas)


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio, turno=TURNO)
    print(f"{len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas")

    Y, unidades, dias = montar_painel(meta, espectro)
    print(f"Painel: {len(unidades)} unidades x {len(dias)} dias "
          f"({', '.join(dias)}) x {len(w)} bandas")
    if np.isnan(Y).any():
        raise SystemExit("Painel incompleto: alguma unidade não tem todos os dias.")

    condicao = np.array([u.split("|")[2] for u in unidades])
    grupos = {"todas": np.ones(len(unidades), dtype=bool)}
    for nivel in sorted(set(condicao)):
        grupos[nivel] = condicao == nivel

    por_banda = []
    posthoc = []

    for grupo, mask_grupo in grupos.items():
        print(f"\nGrupo '{grupo}' -- {int(mask_grupo.sum())} unidades")
        df, df_posthoc = analisar_grupo(Y[mask_grupo], w, dias, grupo)
        por_banda.append(df)
        posthoc.append(df_posthoc)

        print(f"    vias -- ANOVA MR: {(df['via'] == 'anova_mr').sum():4d}  "
              f"Friedman: {(df['via'] == 'friedman').sum():4d}")
        print(f"    bandas com efeito de tempo (q<{ALPHA}): "
              f"{int(df['significativa'].sum())} de {len(df)} "
              f"({df['significativa'].mean():.1%})")
        print(f"    pós-hoc: {int(df_posthoc['significativa'].sum())} de "
              f"{len(df_posthoc)} comparações significativas")

    df_banda = pd.concat(por_banda, ignore_index=True)
    df_posthoc = pd.concat(posthoc, ignore_index=True)
    df_sep = separacao_temporal(df_posthoc)
    df_sensiveis = bandas_sensiveis(df_banda)
    df_resumo = resumir(df_banda, df_posthoc, df_sep)

    df_banda.to_csv(SAIDA_DIR / "temporal_por_banda.csv", sep=";", index=False)
    df_posthoc.to_csv(SAIDA_DIR / "tukey_posthoc.csv", sep=";", index=False)
    df_sep.to_csv(SAIDA_DIR / "separacao_temporal.csv", sep=";", index=False)
    df_sensiveis.to_csv(SAIDA_DIR / "bandas_sensiveis.csv", sep=";", index=False)
    df_resumo.to_csv(SAIDA_DIR / "temporal_resumo.csv", sep=";", index=False)

    print("\nMomento de maior separação espectral (top 3 pares por grupo):")
    for grupo, sub in df_sep.groupby("grupo", sort=True):
        print(f"  {grupo}")
        for _, row in sub.head(3).iterrows():
            marca = " [baseline]" if row["baseline"] else ""
            print(f"    {row['dia_a']} vs {row['dia_b']}{marca:<12} "
                  f"bandas sig: {row['prop_sig']:6.1%}  "
                  f"distancia: {row['dist_euclidiana']:8.3f}")

    print("\nBandas mais sensíveis à evolução temporal (top 5 por grupo):")
    for grupo, sub in df_sensiveis.groupby("grupo", sort=True):
        topo = ", ".join(
            f"{int(r['banda_nm'])} nm ({r['efeito_tempo']:.3f})"
            for _, r in sub.head(5).iterrows()
        )
        print(f"  {grupo:<8} {topo}")

    print(f"\nResultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
