#!/usr/bin/env python3
"""Parâmetros fisiológicos × índices, seguindo o experimento relacaoCRA.

As correlações reúnem genótipos e condições por data. Cada figura fixa um índice
e reúne sete painéis de datas, com pontos circulares e sem linhas de tendência.
Execute:
    python3 parametros_vs_indices/analisar_parametros_vs_indices.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ENTRADA_FISIO = ROOT / "ParametrosFisiologicos.xlsx"
ENTRADA_ESPECTRAL = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
SAIDA = Path(__file__).resolve().parent / "resultados"
PARAMETROS = {
    "CHL A": "chl_a", "CHL B": "chl_b", "CHL TOTAL": "chl_total",
    "Tleaf": "tleaf", "A": "a", "Ci": "ci", "gs": "gs", "E": "e", "VPD": "vpd",
}
COLETAS = {
    "23/02": "D02M", "24/02": "D03M", "25/02": "D04M", "26/02": "D05M",
    "27/02": "D06M", "02/03": "D09M", "03/03": "D10M",
}
GENOTIPOS = ("BR16", "CD202", "EMB48")
CONDICOES = ("IRRIG", "NIRRIG")
CHAVES = ["bloco", "genotipo", "condicao"]
CHAVE_PARCELA = ["data_coleta", *CHAVES]
TODAS_DATAS = "Todas as datas (descritivo)"
MIN_PARES = 8
CORES_CONDICOES = {"IRRIG": "#1976d2", "NIRRIG": "#d32f2f"}
ROTULOS_CONDICOES = {"IRRIG": "Irrigado", "NIRRIG": "Não irrigado"}
INDICES = {
    "PRI": {
        "formula": "(R531 − R570) / (R531 + R570)",
        "bandas": ("531", "570"),
        "calcular": lambda d: (d["531"] - d["570"]) / (d["531"] + d["570"]),
    },
    "NDVI": {
        "formula": "(R800 − R670) / (R800 + R670)",
        "bandas": ("800", "670"),
        "calcular": lambda d: (d["800"] - d["670"]) / (d["800"] + d["670"]),
    },
    "NDRE": {
        "formula": "(R790 − R720) / (R790 + R720)",
        "bandas": ("790", "720"),
        "calcular": lambda d: (d["790"] - d["720"]) / (d["790"] + d["720"]),
    },
    "WDI": {
        "formula": "R970 / R900",
        "bandas": ("970", "900"),
        "calcular": lambda d: d["970"] / d["900"],
    },
    "MSI": {
        "formula": "R1600 / R820",
        "bandas": ("1600", "820"),
        "calcular": lambda d: d["1600"] / d["820"],
    },
}
BANDAS = sorted({b for config in INDICES.values() for b in config["bandas"]}, key=int)
METRICAS = ("pearson_r", "pearson_p", "spearman_rho", "spearman_p",
            "r2_linear", "inclinacao", "intercepto")


def normalizar_cabecalho(nome: object) -> str:
    texto = re.sub(r"\s+", " ", str(nome)).strip()
    return re.sub(r"\((\d{2})[./](\d{2})\)", r"(\1/\2)", texto)


def exigir_colunas(dados: pd.DataFrame, colunas: list[str], origem: str) -> None:
    faltantes = [c for c in colunas if c not in dados.columns]
    if faltantes:
        raise ValueError(f"{origem}: colunas obrigatórias ausentes: {', '.join(faltantes)}")


def numerico(valores: pd.Series) -> pd.Series:
    return pd.to_numeric(
        valores.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)


def exigir_unicidade(dados: pd.DataFrame, chave: list[str], origem: str) -> None:
    repetidos = dados.loc[dados.duplicated(chave, keep=False), chave]
    if not repetidos.empty:
        raise ValueError(f"{origem}: chaves duplicadas: {repetidos.head().to_dict('records')}")


def preparar_fisiologia(fisio: pd.DataFrame) -> pd.DataFrame:
    fisio = fisio.copy()
    fisio.columns = [normalizar_cabecalho(c) for c in fisio.columns]
    if fisio.columns.duplicated().any():
        raise ValueError("Fisiologia: cabeçalhos duplicados após normalização.")
    colunas_medidas = [f"{p} ({data})" for p in PARAMETROS for data in COLETAS]
    exigir_colunas(fisio, ["Genótipo", "Condição", *colunas_medidas], "Fisiologia")
    fisio["linha_excel"] = np.arange(len(fisio)) + 2
    fisio = fisio.dropna(subset=["Genótipo", "Condição", *colunas_medidas], how="all").copy()
    fisio["genotipo"] = fisio["Genótipo"].astype(str).str.strip().str.upper()
    fisio["condicao"] = (fisio["Condição"].astype(str).str.strip().str.upper()
                        .replace({"IRR": "IRRIG", "NIR": "NIRRIG"}))
    if not (fisio["genotipo"].isin(GENOTIPOS) & fisio["condicao"].isin(CONDICOES)).all():
        raise ValueError("Fisiologia: genótipo ou condição ausente/desconhecido.")
    contagens = fisio.groupby(["genotipo", "condicao"]).size()
    esperados = {(g, c) for g in GENOTIPOS for c in CONDICOES}
    if set(contagens.index) != esperados or not contagens.eq(4).all():
        raise ValueError("Fisiologia: esperadas quatro repetições em cada genótipo × condição; "
                         f"observado: {contagens.to_dict()}")
    # A ordem da planilha é a suposição herdada do CRA. Não filtrar medidas antes.
    fisio["bloco"] = "B" + (fisio.groupby(["genotipo", "condicao"], sort=False)
                            .cumcount() + 1).astype(str)
    partes = []
    for parametro in PARAMETROS:
        for data, coleta in COLETAS.items():
            parte = fisio[["linha_excel", *CHAVES]].copy()
            parte["data"] = data
            parte["data_coleta"] = coleta
            parte["parametro"] = parametro
            parte["valor"] = numerico(fisio[f"{parametro} ({data})"])
            partes.append(parte)
    longo = pd.concat(partes, ignore_index=True)
    exigir_unicidade(longo, [*CHAVE_PARCELA, "parametro"], "Fisiologia")
    return longo


def calcular_indices(medias: pd.DataFrame) -> pd.DataFrame:
    dados = medias.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for nome, config in INDICES.items():
            dados[nome] = config["calcular"](dados)
    dados[list(INDICES)] = dados[list(INDICES)].replace([np.inf, -np.inf], np.nan)
    return dados


def preparar_espectros(espectros: pd.DataFrame) -> pd.DataFrame:
    exigir_colunas(espectros, [*CHAVE_PARCELA, "turno", *BANDAS], "Espectros")
    espectros = espectros.copy()
    espectros["turno"] = espectros["turno"].astype(str).str.strip().str.lower()
    espectros["data_coleta"] = espectros["data_coleta"].astype(str).str.strip().str.upper()
    espectros = espectros.loc[espectros["turno"].eq("manha")
                              & espectros["data_coleta"].isin(COLETAS.values())].copy()
    for chave in CHAVES:
        espectros[chave] = espectros[chave].astype(str).str.strip().str.upper()
    if not (espectros["genotipo"].isin(GENOTIPOS)
            & espectros["condicao"].isin(CONDICOES)
            & espectros["bloco"].isin(["B1", "B2", "B3", "B4"])).all():
        raise ValueError("Espectros matinais: genótipo, condição ou bloco inválido.")
    for banda in BANDAS:
        espectros[banda] = numerico(espectros[banda])
    grupos = espectros.groupby(CHAVE_PARCELA, sort=True)
    medias = grupos[BANDAS].mean()
    medias["n_leituras"] = grupos.size()
    medias = medias.join(grupos[BANDAS].count().rename(columns=lambda c: f"n_validos_{c}"))
    medias = calcular_indices(medias.reset_index())
    exigir_unicidade(medias, CHAVE_PARCELA, "Médias espectrais")
    return medias


def parear_dados(fisio: pd.DataFrame, medias: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exigir_unicidade(fisio, [*CHAVE_PARCELA, "parametro"], "Fisiologia longa")
    exigir_unicidade(medias, CHAVE_PARCELA, "Médias espectrais")
    parcelas = fisio[[*CHAVE_PARCELA, "linha_excel"]].drop_duplicates()
    exigir_unicidade(parcelas, CHAVE_PARCELA, "Vínculo entre linha e parcela")
    cobertura = ["n_leituras", *[f"n_validos_{b}" for b in BANDAS]]
    auditoria = parcelas.merge(medias[[*CHAVE_PARCELA, *cobertura]], on=CHAVE_PARCELA,
                               how="outer", validate="one_to_one", indicator=True)
    auditoria["pareamento"] = auditoria.pop("_merge").astype(str).map({
        "both": "pareado", "left_only": "sem_espectro", "right_only": "sem_fisiologia",
    })
    auditoria["data"] = auditoria["data_coleta"].map({v: k for k, v in COLETAS.items()})
    dados = fisio.merge(medias[[*CHAVE_PARCELA, "n_leituras", *INDICES]], on=CHAVE_PARCELA,
                        how="left", validate="many_to_one")
    return dados, auditoria.sort_values(CHAVE_PARCELA).reset_index(drop=True)


def preparar_dados(entrada_fisio: Path = ENTRADA_FISIO,
                   entrada_espectral: Path = ENTRADA_ESPECTRAL
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fisio = preparar_fisiologia(pd.read_excel(entrada_fisio))
    colunas = [*CHAVE_PARCELA, "turno", *BANDAS]
    cabecalho = pd.read_csv(entrada_espectral, sep=";", nrows=0)
    exigir_colunas(cabecalho, colunas, "CSV espectral")
    espectros = pd.read_csv(entrada_espectral, sep=";", usecols=colunas, decimal=",")
    medias = preparar_espectros(espectros)
    dados, auditoria = parear_dados(fisio, medias)
    return dados, medias, auditoria


def pares_validos(dados: pd.DataFrame, indice: str) -> pd.DataFrame:
    return dados.replace({"valor": {np.inf: np.nan, -np.inf: np.nan},
                          indice: {np.inf: np.nan, -np.inf: np.nan}}).dropna(subset=["valor", indice])


def estatisticas_grupo(grupo: pd.DataFrame, indice: str, contexto: str) -> dict:
    pares = pares_validos(grupo, indice)
    n = len(pares)
    if n < MIN_PARES:
        raise ValueError(f"{contexto}: pareamento insuficiente ({n} pares válidos; mínimo {MIN_PARES}).")
    resultado = {"n": n, "n_total": len(grupo), "n_excluidos": len(grupo) - n,
                 "status": "ok", **dict.fromkeys(METRICAS, np.nan)}
    x, y = pares[indice].to_numpy(float), pares["valor"].to_numpy(float)
    constantes = []
    if np.ptp(x) == 0:
        constantes.append("indice_constante")
    if np.ptp(y) == 0:
        constantes.append("parametro_constante")
    if constantes:
        resultado["status"] = ";".join(constantes)
        return resultado
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    ajuste = linregress(x, y)
    resultado.update(pearson_r=r, pearson_p=p_r, spearman_rho=rho, spearman_p=p_rho,
                     r2_linear=ajuste.rvalue ** 2, inclinacao=ajuste.slope,
                     intercepto=ajuste.intercept)
    return resultado


def correlacoes(dados: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for parametro in PARAMETROS:
        parte = dados.loc[dados["parametro"].eq(parametro)]
        grupos = [(data, parte.loc[parte["data"].eq(data)]) for data in COLETAS]
        grupos.append((TODAS_DATAS, parte))
        for data, grupo in grupos:
            for indice, config in INDICES.items():
                linhas.append({
                    "parametro": parametro, "data": data,
                    "tipo_analise": "descritivo" if data == TODAS_DATAS else "por_data",
                    "indice": indice, "formula": config["formula"],
                    **estatisticas_grupo(grupo, indice, f"{parametro} × {indice}, {data}"),
                })
    return pd.DataFrame(linhas)


def salvar_figura(fig, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho.with_suffix(".png"), dpi=200, bbox_inches="tight")
    # Evita timestamps variáveis no PDF ao reproduzir a execução.
    fig.savefig(caminho.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def graficos_agrupados(dados: pd.DataFrame, parametro: str, saida: Path) -> None:
    """Uma figura por índice, com sete datas e genótipos juntos em cada eixo."""
    for indice in INDICES:
        fig, eixos = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey=True)
        for ax, (data, coleta) in zip(eixos.flat, COLETAS.items()):
            por_data = dados.loc[dados["data"].eq(data)]
            for condicao in CONDICOES:
                parte = pares_validos(por_data.loc[por_data["condicao"].eq(condicao)], indice)
                cor = CORES_CONDICOES[condicao]
                ax.scatter(parte[indice], parte["valor"], color=cor, s=35, marker="o", alpha=.9)
            ax.set_title(f"{data} · {coleta}", fontsize=16, fontweight="semibold")
            ax.set_xlabel(indice, fontsize=14)
            ax.set_ylabel(parametro, fontsize=14)
            # Compartilhar as escalas não deve ocultar os valores nos painéis internos.
            ax.tick_params(axis="both", labelbottom=True, labelleft=True, labelsize=12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.get_offset_text().set_fontsize(12)
            ax.yaxis.get_offset_text().set_fontsize(12)
            ax.margins(x=.12, y=.15)
            ax.grid(alpha=.25)
        legenda = eixos.flat[7]
        legenda.axis("off")
        handles = [Line2D([], [], color=CORES_CONDICOES[c], marker="o", linestyle="none",
                          label=ROTULOS_CONDICOES[c]) for c in CONDICOES]
        legenda.legend(handles=handles, loc="center", frameon=False, fontsize=14)
        fig.suptitle(f"{parametro} × {indice}", fontsize=22, fontweight="semibold")
        fig.text(.5, .935, INDICES[indice]["formula"], ha="center", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .91))
        salvar_figura(fig, saida / "agrupados" / PARAMETROS[parametro] / indice.lower())


def remover_graficos_antigos(saida: Path) -> None:
    """Remove somente os nomes gerados pelos formatos substituídos."""
    for slug in PARAMETROS.values():
        for extensao in ("png", "pdf"):
            (saida / "agrupados" / f"{slug}.{extensao}").unlink(missing_ok=True)
            for coleta in COLETAS.values():
                (saida / "agrupados" / slug / f"{coleta.lower()}.{extensao}").unlink(missing_ok=True)
                for indice in INDICES:
                    (saida / "agrupados" / slug
                     / f"{coleta.lower()}_{indice.lower()}.{extensao}").unlink(missing_ok=True)
            for genotipo in GENOTIPOS:
                for indice in INDICES:
                    (saida / "separados" / genotipo
                     / f"{slug}_{indice.lower()}.{extensao}").unlink(missing_ok=True)
    for pasta in [*(saida / "separados" / g for g in GENOTIPOS), saida / "separados"]:
        if pasta.is_dir() and not any(pasta.iterdir()):
            pasta.rmdir()


def grafico_heatmap(tabela: pd.DataFrame, parametro: str, saida: Path) -> None:
    ordem = [*COLETAS, TODAS_DATAS]
    parte = tabela.loc[tabela["parametro"].eq(parametro)]
    matriz = (parte.pivot(index="indice", columns="data", values="spearman_rho")
              .reindex(index=list(INDICES), columns=ordem))
    fig, ax = plt.subplots(figsize=(12, 4.7))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    imagem = ax.imshow(matriz.to_numpy(float), vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(ordem)), [*COLETAS, "Todas as datas\n(descritivo)"])
    ax.set_yticks(range(len(INDICES)), list(INDICES))
    ax.tick_params(axis="both", labelsize=13)
    for i in range(len(INDICES)):
        for j in range(len(ordem)):
            valor = matriz.iat[i, j]
            ax.text(j, i, f"{valor:.2f}" if np.isfinite(valor) else "ND", ha="center", va="center",
                    color="white" if np.isfinite(valor) and abs(valor) > .55 else "black", fontsize=12)
    ax.axvline(len(COLETAS) - .5, color="black", linewidth=1.5)
    ax.set_title(f"{parametro} — correlação de Spearman por data", fontsize=18, fontweight="semibold")
    barra = fig.colorbar(imagem, ax=ax)
    barra.set_label("ρ de Spearman", fontsize=14)
    barra.ax.tick_params(labelsize=12)
    fig.tight_layout()
    salvar_figura(fig, saida / "heatmaps" / PARAMETROS[parametro])


def tabela_markdown(tabela: pd.DataFrame) -> list[str]:
    linhas = ["| Data | Índice | n | ρ Spearman | p Spearman | r Pearson | p Pearson | R² | Status |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in tabela.itertuples(index=False):
        def fmt(valor):
            return f"{valor:.6g}" if np.isfinite(valor) else "ND"
        linhas.append(f"| {r.data} | {r.indice} | {r.n} | {fmt(r.spearman_rho)} | "
                      f"{fmt(r.spearman_p)} | {fmt(r.pearson_r)} | {fmt(r.pearson_p)} | "
                      f"{fmt(r.r2_linear)} | {r.status} |")
    return linhas


def relatorio(dados: pd.DataFrame, medias: pd.DataFrame, auditoria: pd.DataFrame,
              tabela: pd.DataFrame, saida: Path) -> None:
    n_pareados = auditoria["pareamento"].eq("pareado").sum()
    diarios = tabela.loc[tabela["tipo_analise"].eq("por_data")]
    linhas = [
        "# Parâmetros fisiológicos × índices espectrais", "",
        "## Método", "",
        "Adaptação de `relacaoCRA/analisar_indices_cra.py`. Reflectância bruta média das leituras "
        "matinais por parcela (bloco × genótipo × condição × coleta), seguida do cálculo dos índices.", "",
        "As quatro linhas de cada genótipo × condição na planilha são associadas, na ordem, a B1–B4, "
        "antes de excluir medidas ausentes. Essa correspondência é uma suposição herdada do CRA; "
        "o vínculo com a linha original está na auditoria.", "",
        "Cada correlação por data reúne BR16, CD202 e EMB48, irrigados e não irrigados. Pearson "
        "usa `pearsonr`, Spearman usa o p-valor aproximado de `spearmanr`, e a regressão linear "
        "tabulada usa `linregress`. "
        "São apresentados somente p-valores brutos, sem correção para múltiplos testes. "
        "A referência p ≤ 0,05 é exploratória; magnitude e consistência também devem ser avaliadas.", "",
        "O mínimo é de oito pares válidos por comparação. Ausências são excluídas por par "
        "parâmetro–índice; variáveis constantes resultam em métricas indefinidas (ND). "
        "O R² descreve o ajuste nos próprios dados, sem validação preditiva.", "",
        "O agrupamento de todas as datas é apenas descritivo: inclui mudanças temporais e "
        "observações repetidas das mesmas parcelas. As correlações por data também podem refletir "
        "diferenças entre genótipos e condições e não demonstram causalidade.", "",
        "Cada figura de dispersão fixa um parâmetro e um índice e reúne sete painéis de datas. "
        "BR16, CD202 e EMB48 ficam juntos no mesmo eixo, todos com marcadores circulares. "
        "As cores identificam irrigados e não irrigados. Os gráficos mostram somente pontos, "
        "sem linhas de tendência. "
        "As correlações exportadas reúnem as duas condições, totalizando 24 parcelas por data.", "",
        "## Cobertura e auditoria", "",
        f"- Parcelas-data com correspondência: **{n_pareados}**.",
        f"- Valores fisiológicos finitos com espectro pareado: **"
        f"{int((dados['valor'].notna() & dados['n_leituras'].notna()).sum())}**.",
        f"- Comparações nas sete datas: **{len(diarios)}**; agrupadas descritivas: **{len(tabela) - len(diarios)}**.",
        f"- Resultados com status diferente de ok: **{int(tabela['status'].ne('ok').sum())}**.",
        f"- Distribuição do número de leituras por parcela-data: {medias['n_leituras'].value_counts().sort_index().to_dict()}.",
        f"- Situações de pareamento: {auditoria['pareamento'].value_counts().to_dict()}.", "",
        "Todas as leituras disponíveis entram na média. Os totais e as contagens válidas por banda "
        "são registrados; não há descarte automático de parcelas com mais de oito leituras.", "",
        "| Data | Coleta | Pares por comparação (mín.–máx.) |",
        "|---|---|---|",
    ]
    for data, coleta in COLETAS.items():
        ns = diarios.loc[diarios["data"].eq(data), "n"]
        linhas.append(f"| {data} | {coleta} | {ns.min()}–{ns.max()} |")
    linhas += ["", "## Fórmulas", ""]
    for nome, config in INDICES.items():
        linhas.append(f"- **{nome}:** `{config['formula']}`.")
    linhas += ["", "WDI segue a convenção de `indicesEspectrais`: R970/R900. "
               "É a razão inversa do WBI (R900/R970) usado no CRA.", "", "## Arquivos", "",
               "- [Dados pareados](dados_pareados.csv)",
               "- [Estatísticas e p-valores brutos](correlacoes.csv)",
               "- [Auditoria do pareamento e contagens de leituras](auditoria_pareamento.csv)",
               "- [Reflectâncias médias e índices por parcela](medias_espectrais_indices.csv)", ""]
    for parametro, slug in PARAMETROS.items():
        linhas += [f"## {parametro}", "",
                   f"[Mapa de Spearman](heatmaps/{slug}.png) · [PDF](heatmaps/{slug}.pdf)", "",
                   *tabela_markdown(tabela.loc[tabela["parametro"].eq(parametro)]), "",
                   "Uma imagem por índice, com sete painéis de datas e genótipos juntos em cada eixo:", "",
                   "| Índice | Imagem | Documento |", "|---|---|---|"]
        for indice in INDICES:
            linhas.append(f"| {indice} | [PNG](agrupados/{slug}/{indice.lower()}.png) | "
                          f"[PDF](agrupados/{slug}/{indice.lower()}.pdf) |")
        linhas.append("")
    linhas += ["## Execução", "", f"Python {sys.version.split()[0]}; pandas {pd.__version__}; "
               f"NumPy {np.__version__}; Matplotlib {matplotlib.__version__}.", "",
               "Gráficos: 45 figuras por parâmetro × índice, com sete datas e genótipos no mesmo eixo, "
               "e 9 mapas de calor; "
               "cada figura é exportada em PNG e PDF."]
    (saida / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    print("Preparando fisiologia, médias espectrais e pareamento...", flush=True)
    dados, medias, auditoria = preparar_dados()
    tabela = correlacoes(dados)
    SAIDA.mkdir(parents=True, exist_ok=True)
    for nome, frame in (("dados_pareados", dados), ("medias_espectrais_indices", medias),
                        ("auditoria_pareamento", auditoria), ("correlacoes", tabela)):
        frame.to_csv(SAIDA / f"{nome}.csv", sep=";", index=False, float_format="%.12g")
    remover_graficos_antigos(SAIDA)
    plt.style.use("seaborn-v0_8-whitegrid")
    for parametro in PARAMETROS:
        parte = dados.loc[dados["parametro"].eq(parametro)]
        graficos_agrupados(parte, parametro, SAIDA)
        grafico_heatmap(tabela, parametro, SAIDA)
        print(f"{parametro}: cinco figuras por índice e um mapa exportados em PNG/PDF.", flush=True)
    relatorio(dados, medias, auditoria, tabela, SAIDA)
    print(f"Concluído: {len(medias)} parcelas-data, {len(dados)} registros fisiológicos, "
          f"{len(tabela)} comparações. Resultados em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
