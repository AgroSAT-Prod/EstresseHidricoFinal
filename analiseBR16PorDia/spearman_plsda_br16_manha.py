#!/usr/bin/env python3
"""BR16 pela manhã: Spearman + PLS-DA, separadamente para cada dia.

Para cada dia, as bandas contíguas com |rho de Spearman| > 0,80 são
condensadas (ligação completa, janela máxima de 10 nm). A representante de
cada grupo é a banda com menor p do contraste BR16 IRRIG x NIRRIG naquele dia.
Em seguida, um PLS-DA é ajustado somente com essas representantes. O número
de componentes é escolhido por validação leave-one-block-out e as cinco bandas
significativas (q_FDR <= 0,05) com maior VIP são o resultado principal.

Os p/q e o delta de Cliff são reaproveitados da comparação não paramétrica já
existente, que foi calculada nas leituras do turno da manhã. A validação do
PLS-DA, porém, mantém as oito leituras de uma parcela juntas pela coluna
``bloco`` para evitar vazamento entre treino e teste.

Uso:
    .venv/bin/python analiseBR16PorDia/spearman_plsda_br16_manha.py
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "testeDeNormalidade"))
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))

from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import agrupar, spearman_matriz  # noqa: E402

SAIDA = ROOT / "analiseBR16PorDia" / "dataset_gerado"
COMPARACAO = (
    ROOT / "testeDiferencaSignificativa" / "dataset_gerado" / "comparacao_estresse.csv"
)

TURNO = "manha"
ESTAGIO = "normalizado"
ALPHA = 0.05
TOP_K = 5
MAX_COMPONENTES = 10


def escolher_representantes(
    corr: np.ndarray, w: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa com Spearman 0,80 e escolhe por menor p dentro de cada grupo."""
    grupos = agrupar(corr, w)
    reps = np.zeros(len(w), dtype=bool)
    for grupo in np.unique(grupos):
        idx = np.flatnonzero(grupos == grupo)
        # Desempate pelo medoide, para que a escolha não dependa da ordem.
        melhor_p = np.nanmin(p[idx])
        candidatos = idx[np.isclose(p[idx], melhor_p)]
        if len(candidatos) > 1:
            centralidade = np.abs(corr[np.ix_(candidatos, idx)]).mean(axis=1)
            escolhido = candidatos[np.argmax(centralidade)]
        else:
            escolhido = candidatos[0]
        reps[escolhido] = True
    return grupos, reps


def escolher_componentes(X: np.ndarray, y: np.ndarray, blocos: np.ndarray) -> tuple[int, pd.DataFrame]:
    """Escolhe componentes por balanced accuracy em leave-one-block-out."""
    cv = GroupKFold(n_splits=len(np.unique(blocos)))
    dobras = list(cv.split(X, y, groups=blocos))
    limite = min(MAX_COMPONENTES, X.shape[1], min(len(treino) - 1 for treino, _ in dobras))
    linhas = []
    for n_comp in range(1, limite + 1):
        scores = []
        for treino, teste in dobras:
            pls = PLSRegression(n_components=n_comp, scale=True)
            pls.fit(X[treino], y[treino])
            pred = (pls.predict(X[teste]).ravel() >= 0).astype(int)
            scores.append(balanced_accuracy_score(y[teste], pred))
        linhas.append({
            "n_componentes": n_comp,
            "balanced_accuracy_cv": float(np.mean(scores)),
            "desvio_cv": float(np.std(scores)),
        })
    validacao = pd.DataFrame(linhas)
    melhor = int(validacao.loc[validacao["balanced_accuracy_cv"].idxmax(), "n_componentes"])
    return melhor, validacao


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """Variable Importance in Projection para um PLS-DA de resposta única."""
    t, pesos, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ssy = (q ** 2).ravel() * (t ** 2).sum(axis=0)
    pesos = pesos / np.linalg.norm(pesos, axis=0, keepdims=True)
    return np.sqrt(pesos.shape[0] * ((pesos ** 2) * ssy).sum(axis=1) / ssy.sum())


def analisar_dia(
    dia: str, meta: pd.DataFrame, espectro: np.ndarray, w: np.ndarray, estat: pd.DataFrame,
    saida_base: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    estat = estat.set_index("banda_nm").reindex(w.astype(int)).reset_index()
    p = estat["p_valor"].to_numpy()
    corr = spearman_matriz(espectro)
    grupos, reps = escolher_representantes(corr, w, p)
    X = espectro[:, reps]
    bandas = w[reps].astype(int)
    y = (meta["condicao"].to_numpy() == "NIRRIG").astype(int)

    n_comp, validacao = escolher_componentes(X, y, meta["bloco"].to_numpy())
    pls = PLSRegression(n_components=n_comp, scale=True).fit(X, y)
    resultado = estat.loc[reps, ["banda_nm", "p_valor", "q_fdr", "epsilon2", "delta_cliff", "significativa"]].copy()
    resultado.insert(0, "dia", dia)
    resultado["grupo_spearman"] = grupos[reps]
    resultado["vip_pls_da"] = vip_scores(pls)
    resultado["vip_acima_de_1"] = resultado["vip_pls_da"] > 1
    resultado["abs_delta_cliff"] = resultado["delta_cliff"].abs()

    # "Mais significativas" exige significância estatística e importância no
    # discriminante: entre q <= 0,05, o VIP maior determina a ordem.
    top = resultado[resultado["significativa"]].nlargest(TOP_K, "vip_pls_da").copy()
    top.insert(1, "posicao", np.arange(1, len(top) + 1))

    pasta = saida_base / dia
    pasta.mkdir(parents=True, exist_ok=True)
    validacao.to_csv(pasta / "pls_da_validacao.csv", sep=";", index=False)
    resultado.sort_values("vip_pls_da", ascending=False).to_csv(
        pasta / "bandas_representativas_pls_da.csv", sep=";", index=False
    )
    top.to_csv(pasta / "top5_bandas.csv", sep=";", index=False)

    resumo = {
        "dia": dia, "amostras": len(meta), "bandas_originais": len(w),
        "grupos_spearman": int(grupos.max()) + 1,
        "reducao": 1 - (int(grupos.max()) + 1) / len(w),
        "limiar_spearman_abs_rho": 0.80, "janela_nm": 10,
        "n_componentes_pls_da": n_comp,
        "balanced_accuracy_cv": validacao.loc[validacao["n_componentes"] == n_comp, "balanced_accuracy_cv"].iloc[0],
        "bandas_significativas_representativas": int(resultado["significativa"].sum()),
        "top5_encontradas": len(top),
    }
    return top, resumo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spearman + PLS-DA por dia e genótipo, turno da manhã."
    )
    parser.add_argument(
        "--genotipo", default="BR16", choices=("BR16", "EMB48", "CD202"),
        help="Genótipo a analisar (padrão: BR16).",
    )
    args = parser.parse_args()
    genotipo = args.genotipo
    if not COMPARACAO.exists():
        raise SystemExit(f"Arquivo de significância não encontrado: {COMPARACAO}")
    meta, espectro, w = carregar(ESTAGIO, turno=TURNO)
    mask = meta["genotipo"].eq(genotipo).to_numpy()
    meta, espectro = meta.loc[mask].reset_index(drop=True), espectro[mask]
    estatisticas = pd.read_csv(COMPARACAO, sep=";")
    estatisticas = estatisticas[estatisticas["genotipo"].eq(genotipo)]

    tops, resumos = [], []
    saida_genotipo = SAIDA / genotipo
    for dia in sorted(meta["dia"].unique()):
        mascara_dia = meta["dia"].eq(dia).to_numpy()
        top, resumo = analisar_dia(
            dia, meta.loc[mascara_dia].reset_index(drop=True), espectro[mascara_dia], w,
            estatisticas[estatisticas["dia"].eq(dia)], saida_genotipo,
        )
        tops.append(top)
        resumos.append(resumo)
        print(f"{dia}: {resumo['grupos_spearman']} representantes; "
              f"PLS-DA {resumo['n_componentes_pls_da']} comp.; "
              f"BAC-CV {resumo['balanced_accuracy_cv']:.3f}; top {len(top)}")

    saida_genotipo.mkdir(parents=True, exist_ok=True)
    pd.concat(tops, ignore_index=True).to_csv(
        saida_genotipo / f"top5_bandas_{genotipo}_manha_por_dia.csv", sep=";", index=False
    )
    pd.DataFrame(resumos).to_csv(saida_genotipo / "resumo_por_dia.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
