#!/usr/bin/env python3
"""Interacao genotipo x condicao por ART, e os efeitos simples que a decompoem.

Preenche a lacuna que `diferencas_significativas.py` admite em `INTERACAO_KW`:
Kruskal-Wallis e um teste de um fator e nao tem termo de interacao, entao a
coluna `sig_interacao` do resumo daquele modulo fica vazia em todos os dias.
Sem esse termo nao da para responder a pergunta pratica -- posso agrupar os
genotipos, ou a resposta ao estresse depende do material?

ART (Aligned Rank Transform)
----------------------------
Para testar um termo, remove-se de cada observacao tudo o que nao e aquele
termo, e so entao se posteia. Para a interacao:

    alinhado = (y - media da celula) + (media da celula - media A - media B + geral)

O primeiro parenteses e o residuo, o segundo e o efeito de interacao puro. Os
postos desse alinhado entram numa ANOVA fatorial comum, e o F do termo de
interacao e valido sob nao-normalidade. Wobbrock et al. (2011).

O que o script produz, dia a dia e banda a banda
------------------------------------------------
1. ART completo nos 3 genotipos x 2 condicoes: genotipo, condicao, interacao.
2. ART par a par (2x2), que localiza qual genotipo carrega a interacao.
3. Efeitos simples: dentro de IRRIG e dentro de NIRRIG separadamente, os dois
   genotipos do par diferem? E a decomposicao que torna a decisao acionavel --

   | interacao | efeito simples | leitura |
   |-----------|----------------|---------|
   | nao       | nao            | agrupavel |
   | nao       | sim            | mesma resposta ao estresse, niveis diferentes |
   | sim       | qualquer       | nao agrupavel: a resposta depende do genotipo |

Todos os p-valores recebem FDR de Benjamini-Hochberg, com as bandas como
familia. Apenas o turno da manha entra.

Uso:
    python interacao_genotipo_condicao.py [recortado|suavizado|normalizado]
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "preprocessamento_espectral"))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar  # noqa: E402
from selecao_estatistica import fdr_bh  # noqa: E402
from diferencas_significativas import (  # noqa: E402
    CONDICOES,
    GENOTIPOS,
    correcao_empates,
    dummies,
    kruskal_vetorizado,
    rss_multiplo,
)
from comparacao_genotipo_condicao import cliff_delta, epsilon_quadrado  # noqa: E402

SAIDA_DIR = ROOT / "dataset_gerado"

ESTAGIO_PADRAO = "normalizado"

TURNO = "manha"

# O veredito de agrupamento depende de onde se corta a significancia, entao a
# analise roda nos tres niveis e reporta os tres lado a lado. Nao ha custo:
# `fdr_bh` devolve o q-valor ajustado de Benjamini-Hochberg, que nao e funcao
# de alfa -- alfa so entra na comparacao `q < alfa`. Os tres niveis saem do
# mesmo ajuste, sem refazer ART nem Kruskal-Wallis, e sao comparaveis entre si.
ALPHAS = (0.05, 0.01, 0.001)
ALPHA_REFERENCIA = 0.05

# Teste que decide os efeitos simples. As duas opcoes sao calculadas e
# guardadas sempre; esta constante escolhe qual delas move as colunas de
# significancia, o veredito e as figuras.
#
# Kruskal-Wallis com 2 grupos e algebricamente identico ao Mann-Whitney
# bilateral assintotico SEM correcao de continuidade: H = z^2, e
# chi2.sf(H, 1) = 2 * norm.sf(|z|). Verificado nos dados -- os p-valores batem
# ate a ultima casa. O Mann-Whitney exato, porem, e outro teste: com n = 32+32
# e sem empates ele enumera a distribuicao nula em vez de aproxima-la pela
# normal, e diverge na cauda, onde vive a maioria das bandas deste
# experimento.
TESTE_EFEITO_SIMPLES = "mannwhitney"

# Triagem pratica para o veredito de agrupamento: acima desta fracao de bandas
# a diferenca deixa de ser pontual e passa a valer para o espectro como um
# todo. Nao e um teste -- e o corte de leitura da tabela.
LIMIAR_AGRUPAR = 0.20


def coluna_sig(prefixo: str, alpha: float) -> str:
    """Nome da coluna booleana de significancia para um nivel."""
    return f"sig_{prefixo}_{alpha:g}"


def anova_fatorial(
    Y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    niveis_a: list[str],
    niveis_b: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ANOVA de dois fatores com interacao, com os niveis passados explicitamente.

    Mesma decomposicao de somas de quadrados tipo II de
    `diferencas_significativas.anova_dois_fatores`, mas sem depender das
    constantes de modulo daquele arquivo -- os testes par a par precisam de
    fatores de dois niveis, nao dos tres genotipos.
    """
    n = len(Y)
    um = np.ones((n, 1))
    ga = dummies(a, niveis_a)
    gb = dummies(b, niveis_b)
    gab = np.column_stack([ga[:, [i]] * gb for i in range(ga.shape[1])])

    X_completo = np.column_stack([um, ga, gb, gab])
    X_aditivo = np.column_stack([um, ga, gb])
    X_sem_a = np.column_stack([um, gb])
    X_sem_b = np.column_stack([um, ga])

    rss_completo = rss_multiplo(X_completo, Y)
    rss_aditivo = rss_multiplo(X_aditivo, Y)

    gl_residuo = n - X_completo.shape[1]
    qm_residuo = rss_completo / gl_residuo

    termos = {
        "genotipo": (rss_multiplo(X_sem_a, Y) - rss_aditivo, ga.shape[1]),
        "condicao": (rss_multiplo(X_sem_b, Y) - rss_aditivo, gb.shape[1]),
        "interacao": (rss_aditivo - rss_completo, gab.shape[1]),
    }

    resultado = {}
    for termo, (soma_quadrados, gl) in termos.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (soma_quadrados / gl) / qm_residuo
        p = f_dist.sf(F, gl, gl_residuo)
        resultado[termo] = (F, np.where(np.isfinite(F), p, np.nan))
    return resultado


def alinhar_interacao(
    Y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    niveis_a: list[str],
    niveis_b: list[str],
) -> np.ndarray:
    """Alinhamento do ART para o termo de interacao."""
    geral = Y.mean(axis=0)
    media_a = {nivel: Y[a == nivel].mean(axis=0) for nivel in niveis_a}
    media_b = {nivel: Y[b == nivel].mean(axis=0) for nivel in niveis_b}

    alinhado = np.zeros_like(Y)
    for nivel_a in niveis_a:
        for nivel_b in niveis_b:
            mask = (a == nivel_a) & (b == nivel_b)
            if not mask.any():
                continue
            media_celula = Y[mask].mean(axis=0)
            efeito = media_celula - media_a[nivel_a] - media_b[nivel_b] + geral
            alinhado[mask] = (Y[mask] - media_celula) + efeito
    return alinhado


def art_termos(
    Y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    niveis_a: list[str],
    niveis_b: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """F e p dos tres termos, cada um sobre seu proprio alinhamento.

    Os efeitos principais saem dos postos brutos -- alinhar para um efeito
    principal num delineamento balanceado nao muda o teste --, e a interacao
    sai do alinhamento proprio, que e onde o ART faz diferenca.
    """
    postos = rankdata(Y, axis=0)
    principais = anova_fatorial(postos, a, b, niveis_a, niveis_b)

    alinhado = alinhar_interacao(Y, a, b, niveis_a, niveis_b)
    interacao = anova_fatorial(
        rankdata(alinhado, axis=0), a, b, niveis_a, niveis_b
    )["interacao"]

    return {
        "genotipo": principais["genotipo"],
        "condicao": principais["condicao"],
        "interacao": interacao,
    }


def mannwhitney_por_banda(
    Y: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """U e p bilateral do Mann-Whitney, banda a banda.

    Usa o metodo exato -- que enumera a distribuicao nula de U em vez de
    aproxima-la pela normal -- sempre que a banda nao tem empates, que e o
    caso geral em espectro continuo depois do SNV. Havendo empate, o exato
    deixa de ser valido e a banda cai para a aproximacao assintotica.
    """
    a, b = Y[idx_a], Y[idx_b]
    n_total = len(idx_a) + len(idx_b)

    U = np.full(Y.shape[1], np.nan)
    p = np.full(Y.shape[1], np.nan)

    for j in range(Y.shape[1]):
        x, y = a[:, j], b[:, j]
        tem_empate = len(np.unique(np.concatenate([x, y]))) < n_total
        resultado = mannwhitneyu(
            x, y, alternative="two-sided",
            method="asymptotic" if tem_empate else "exact",
        )
        U[j] = resultado.statistic
        p[j] = resultado.pvalue

    return U, p


def contraste_celulas(
    Y_sub: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> pd.DataFrame:
    """Contraste de duas amostras banda a banda, ja recortado nas duas celulas.

    `Y_sub` precisa conter so as amostras das duas celulas comparadas: os
    postos e o delta de Cliff so valem se a ordenacao nao incluir nenhuma
    amostra de fora do contraste. Kruskal-Wallis e Mann-Whitney exato saem os
    dois, com as bandas como familia de FDR; `TESTE_EFEITO_SIMPLES` decide qual
    deles move `p_valor`, `q_fdr` e as colunas de significancia.
    """
    postos = rankdata(Y_sub, axis=0)
    empates = correcao_empates(Y_sub)

    H, p_kw = kruskal_vetorizado(postos, [idx_a, idx_b], empates)
    q_kw = np.clip(fdr_bh(p_kw), 0, 1)

    U, p_mw = mannwhitney_por_banda(Y_sub, idx_a, idx_b)
    q_mw = np.clip(fdr_bh(p_mw), 0, 1)

    ativo = "mannwhitney" if TESTE_EFEITO_SIMPLES == "mannwhitney" else "kruskal"
    p_ativo, q_ativo = (p_mw, q_mw) if ativo == "mannwhitney" else (p_kw, q_kw)

    df = pd.DataFrame({
        "n_a": len(idx_a),
        "n_b": len(idx_b),
        "teste": ativo,
        "H": H,
        "p_kruskal": p_kw,
        "q_kruskal": q_kw,
        "U": U,
        "p_mannwhitney": p_mw,
        "q_mannwhitney": q_mw,
        # Colunas do teste ativo, que e o que move as figuras e o veredito.
        "p_valor": p_ativo,
        "q_fdr": q_ativo,
        "epsilon2": epsilon_quadrado(H, len(Y_sub), 2),
        # Positivo: a celula `a` com valores acima da celula `b`.
        "delta_cliff": cliff_delta(postos, idx_a, idx_b),
    })
    for alpha in ALPHAS:
        df[coluna_sig("efeito", alpha)] = q_ativo < alpha
    return df


def efeito_simples(
    Y: np.ndarray,
    genotipo: np.ndarray,
    condicao: np.ndarray,
    par: tuple[str, str],
    nivel_condicao: str,
) -> pd.DataFrame | None:
    """Os dois genotipos do par diferem DENTRO de uma condicao?

    Kruskal-Wallis de duas amostras com os postos recalculados dentro do
    recorte: o contraste e entre os dois materiais naquela condicao, e incluir
    a outra condicao na ordenacao contaminaria o teste e o delta de Cliff.
    """
    a, b = par
    mask = (condicao == nivel_condicao) & np.isin(genotipo, par)
    if mask.sum() < 4:
        return None

    gen_sub = genotipo[mask]
    idx_a = (gen_sub == a).nonzero()[0]
    idx_b = (gen_sub == b).nonzero()[0]
    if len(idx_a) < 2 or len(idx_b) < 2:
        return None

    return contraste_celulas(Y[mask], idx_a, idx_b)


SIGLA_VEREDITO = {
    "agrupavel": "OK", "offset constante": "offset", "nao agrupavel": "NAO",
}


def veredito(prop_interacao: float, prop_irrig: float, prop_nirrig: float) -> str:
    """Traduz as tres proporcoes na decisao de agrupamento."""
    if prop_interacao > LIMIAR_AGRUPAR:
        return "nao agrupavel"
    if max(prop_irrig, prop_nirrig) > LIMIAR_AGRUPAR:
        return "offset constante"
    return "agrupavel"


def analisar_dia(
    Y: np.ndarray,
    w: np.ndarray,
    meta_dia: pd.DataFrame,
    dia: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ART completo, ART par a par e efeitos simples de um dia."""
    genotipo = meta_dia["genotipo"].to_numpy()
    condicao = meta_dia["condicao"].to_numpy()
    bandas = w.astype(int)

    completo = art_termos(Y, genotipo, condicao, GENOTIPOS, CONDICOES)
    df_completo = pd.DataFrame({"dia": dia, "banda_nm": bandas, "n": len(Y)})
    for termo, (F, p) in completo.items():
        q = np.clip(fdr_bh(p), 0, 1)
        df_completo[f"F_{termo}"] = F
        df_completo[f"p_{termo}"] = p
        df_completo[f"q_{termo}"] = q
        for alpha in ALPHAS:
            df_completo[coluna_sig(termo, alpha)] = q < alpha

    linhas_pares, linhas_simples = [], []
    for par in combinations(GENOTIPOS, 2):
        mask = np.isin(genotipo, par)
        termos = art_termos(
            Y[mask], genotipo[mask], condicao[mask], list(par), CONDICOES
        )
        F, p = termos["interacao"]
        q = np.clip(fdr_bh(p), 0, 1)
        df_par = pd.DataFrame({
            "dia": dia,
            "genotipo_a": par[0],
            "genotipo_b": par[1],
            "banda_nm": bandas,
            "n": int(mask.sum()),
            "F_interacao": F,
            "p_interacao": p,
            "q_interacao": q,
        })
        for alpha in ALPHAS:
            df_par[coluna_sig("interacao", alpha)] = q < alpha
        linhas_pares.append(df_par)

        for nivel in CONDICOES:
            simples = efeito_simples(Y, genotipo, condicao, par, nivel)
            if simples is None:
                continue
            simples.insert(0, "banda_nm", bandas)
            simples.insert(0, "condicao", nivel)
            simples.insert(0, "genotipo_b", par[1])
            simples.insert(0, "genotipo_a", par[0])
            simples.insert(0, "dia", dia)
            linhas_simples.append(simples)

    return (
        df_completo,
        pd.concat(linhas_pares, ignore_index=True),
        pd.concat(linhas_simples, ignore_index=True),
    )


def resumir(pares: pd.DataFrame, simples: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dia, par e nivel de significancia."""
    chaves = ["dia", "genotipo_a", "genotipo_b"]
    partes = []

    for alpha in ALPHAS:
        interacao = (
            pares.groupby(chaves)[coluna_sig("interacao", alpha)]
            .mean().rename("prop_interacao")
        )
        por_condicao = (
            simples.groupby(chaves + ["condicao"])
            .agg(prop_sig=(coluna_sig("efeito", alpha), "mean"),
                 delta_mediano=("delta_cliff", "median"))
        )

        df = interacao.reset_index()
        df.insert(0, "alpha", alpha)
        for nivel in CONDICOES:
            sub = por_condicao.xs(nivel, level="condicao")
            df = df.merge(
                sub.reset_index().rename(columns={
                    "prop_sig": f"prop_sig_{nivel}",
                    "delta_mediano": f"delta_{nivel}",
                }),
                on=chaves, how="left",
            )

        df["veredito"] = [
            veredito(r.prop_interacao, r.prop_sig_IRRIG, r.prop_sig_NIRRIG)
            for r in df.itertuples()
        ]
        partes.append(df)

    df = pd.concat(partes, ignore_index=True)
    verificar_monotonicidade(df)
    return df.sort_values(
        ["genotipo_a", "genotipo_b", "dia", "alpha"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def verificar_monotonicidade(resumo: pd.DataFrame) -> None:
    """Um limiar mais rigoroso nao pode declarar mais bandas significativas.

    A checagem e barata e pega de imediato qualquer erro de indexacao entre os
    niveis -- o tipo de bug que passaria despercebido numa tabela de
    porcentagens plausiveis.
    """
    colunas = ["prop_interacao"] + [f"prop_sig_{n}" for n in CONDICOES]
    for chave, sub in resumo.groupby(["dia", "genotipo_a", "genotipo_b"]):
        sub = sub.sort_values("alpha", ascending=False)
        for coluna in colunas:
            valores = sub[coluna].to_numpy()
            if np.any(np.diff(valores) > 1e-12):
                raise AssertionError(
                    f"Monotonicidade violada em {chave}, coluna {coluna}: "
                    f"{dict(zip(sub['alpha'], valores))}"
                )


def comparar_testes(simples: pd.DataFrame) -> None:
    """Quanto Kruskal-Wallis e Mann-Whitney exato discordam."""
    print("\n" + "=" * 96)
    print(f"Kruskal-Wallis vs Mann-Whitney exato  (teste ativo: "
          f"{TESTE_EFEITO_SIMPLES})")
    print("=" * 96)

    razao = simples["p_mannwhitney"] / simples["p_kruskal"].replace(0, np.nan)
    print(f"  Razao p_MW / p_KW: mediana {razao.median():.3f}, "
          f"1o quartil {razao.quantile(0.25):.3f}, "
          f"3o quartil {razao.quantile(0.75):.3f}")
    print(f"  Bandas em que o MW exato da p MENOR: "
          f"{(razao < 1).mean():.1%}")

    print("\n  Bandas que mudam de decisao entre os dois testes:")
    for alpha in ALPHAS:
        divergem = (simples["q_kruskal"] < alpha) ^ (simples["q_mannwhitney"] < alpha)
        so_mw = ((simples["q_mannwhitney"] < alpha)
                 & ~(simples["q_kruskal"] < alpha)).sum()
        so_kw = ((simples["q_kruskal"] < alpha)
                 & ~(simples["q_mannwhitney"] < alpha)).sum()
        print(f"    q < {alpha:<7g} {int(divergem.sum()):6d} de {len(simples)} "
              f"({divergem.mean():5.2%})   so MW: {int(so_mw):5d}   "
              f"so KW: {int(so_kw):5d}")


def main() -> None:
    estagio = sys.argv[1] if len(sys.argv) > 1 else ESTAGIO_PADRAO

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Estágio de pré-processamento: {estagio}")
    meta, espectro, w = carregar(estagio, turno=TURNO)
    print(f"{len(meta)} amostras do turno '{TURNO}' x {len(w)} bandas\n")

    completos, pares, simples = [], [], []
    for dia in sorted(meta["dia"].unique()):
        mask = (meta["dia"] == dia).to_numpy()
        df_c, df_p, df_s = analisar_dia(
            espectro[mask], w, meta[mask].reset_index(drop=True), dia
        )
        completos.append(df_c)
        pares.append(df_p)
        simples.append(df_s)

        print(f"{dia} (n={int(mask.sum())}) -- ART nos 3 genotipos x 2 condicoes:")
        print(f"    {'termo':<10}" + "".join(f"{f'q<{a:g}':>12}" for a in ALPHAS))
        for termo in ("genotipo", "condicao", "interacao"):
            valores = "".join(
                f"{df_c[coluna_sig(termo, a)].mean():>11.1%} " for a in ALPHAS
            )
            print(f"    {termo:<10}{valores}")

    df_completo = pd.concat(completos, ignore_index=True)
    df_pares = pd.concat(pares, ignore_index=True)
    df_simples = pd.concat(simples, ignore_index=True)
    df_resumo = resumir(df_pares, df_simples)

    df_completo.to_csv(SAIDA_DIR / "interacao_por_banda.csv", sep=";", index=False)
    df_pares.to_csv(SAIDA_DIR / "interacao_pares.csv", sep=";", index=False)
    df_simples.to_csv(SAIDA_DIR / "efeitos_simples.csv", sep=";", index=False)
    df_resumo.to_csv(SAIDA_DIR / "interacao_resumo.csv", sep=";", index=False)

    print("\n" + "=" * 96)
    print("Interacao par a par e efeitos simples, nos tres niveis (%)")
    print("=" * 96)
    for (a, b), sub in df_resumo.groupby(["genotipo_a", "genotipo_b"], sort=True):
        print(f"\n{a} vs {b}")
        cabecalho = "".join(f"{f'q<{al:g}':>26}" for al in ALPHAS)
        print(f"  {'dia':<5}{cabecalho}")
        print(f"  {'':<5}" + "".join(
            f"{'int':>7}{'IRR':>6}{'NIR':>6}{'':>7}" for _ in ALPHAS
        ))
        for dia in sorted(sub["dia"].unique()):
            linha = f"  {dia:<5}"
            for al in ALPHAS:
                r = sub[(sub["dia"] == dia) & (sub["alpha"] == al)].iloc[0]
                linha += (f"{r.prop_interacao:>6.0%}{r.prop_sig_IRRIG:>6.0%}"
                          f"{r.prop_sig_NIRRIG:>6.0%}"
                          f"  {SIGLA_VEREDITO[r.veredito]:<6}")
            print(linha)

    print("\n" + "=" * 96)
    print("Veredito por par e nivel (dias em cada categoria):")
    tabela = (
        df_resumo.groupby(["genotipo_a", "genotipo_b", "alpha", "veredito"])
        .size().unstack(fill_value=0)
    )
    print(tabela.to_string())

    print("\nEstabilidade -- dias em que o veredito e o mesmo nos tres niveis:")
    for (a, b), sub in df_resumo.groupby(["genotipo_a", "genotipo_b"], sort=True):
        por_dia = sub.groupby("dia")["veredito"].nunique()
        estaveis = sorted(por_dia[por_dia == 1].index)
        instaveis = sorted(por_dia[por_dia > 1].index)
        print(f"  {a} vs {b:<8} {len(estaveis)}/7 estaveis"
              + (f"   muda em: {', '.join(instaveis)}" if instaveis else ""))

    comparar_testes(df_simples)

    print(f"\nInteracao no modelo completo, por nivel:")
    for alpha in ALPHAS:
        n = int(df_completo[coluna_sig("interacao", alpha)].sum())
        print(f"    q < {alpha:<7g} {n:6d} de {len(df_completo)} testes "
              f"banda-dia ({n / len(df_completo):6.1%})")

    print(f"\nResultados salvos em {SAIDA_DIR}")


if __name__ == "__main__":
    main()
