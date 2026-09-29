#!/usr/bin/env python3
"""Relaciona R680 às clorofilas A, B e total por dia experimental.

Como a planilha fisiológica não possui identificadores de planta/bloco,
a unidade de comparação é a média de cada combinação genótipo × condição
(3 genótipos × 2 condições = 6 observações por dia).

Mapeamento cronológico: D02 -> 23/02, D06 -> 27/02 e D09 -> 02/03.
As leituras espectrais da manhã são usadas para manter o padrão das análises
espectrais por dia existentes neste projeto.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ESPECTRAL = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
FISIOLOGICO = ROOT / "dataset" / "Fisiologicos.xlsx"
SAIDA = Path(__file__).resolve().parent / "dataset_gerado"
DIAS = {"D02": "23/02", "D06": "27/02", "D09": "02/03"}
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def indice_coluna(ref: str) -> int:
    numero = 0
    for letra in re.match(r"[A-Z]+", ref).group():
        numero = numero * 26 + ord(letra) - 64
    return numero - 1


def ler_fisiologico() -> list[list[str]]:
    with ZipFile(FISIOLOGICO) as arquivo:
        strings = [
            "".join(no.text or "" for no in item.findall(".//x:t", NS))
            for item in ET.fromstring(arquivo.read("xl/sharedStrings.xml")).findall("x:si", NS)
        ]
        planilha = ET.fromstring(arquivo.read("xl/worksheets/sheet1.xml"))
        linhas = []
        for linha in planilha.findall(".//x:sheetData/x:row", NS):
            celulas = {}
            for celula in linha.findall("x:c", NS):
                valor = celula.find("x:v", NS)
                texto = "" if valor is None else valor.text
                if celula.get("t") == "s" and texto:
                    texto = strings[int(texto)]
                celulas[indice_coluna(celula.attrib["r"])] = texto
            linhas.append([celulas.get(i, "") for i in range(86)])
    return linhas


def numero(valor: str) -> float:
    return float(valor.replace(",", "."))


def correlacao_pearson(x: list[float], y: list[float]) -> float:
    media_x, media_y = mean(x), mean(y)
    numerador = sum((a - media_x) * (b - media_y) for a, b in zip(x, y))
    denominador = math.sqrt(
        sum((a - media_x) ** 2 for a in x) * sum((b - media_y) ** 2 for b in y)
    )
    if not denominador:
        return float("nan")
    return max(-1.0, min(1.0, numerador / denominador))


def postos(valores: list[float]) -> list[float]:
    ordem = sorted(range(len(valores)), key=valores.__getitem__)
    resultado = [0.0] * len(valores)
    inicio = 0
    while inicio < len(ordem):
        fim = inicio
        while fim + 1 < len(ordem) and valores[ordem[fim + 1]] == valores[ordem[inicio]]:
            fim += 1
        posto = (inicio + fim + 2) / 2
        for posicao in ordem[inicio : fim + 1]:
            resultado[posicao] = posto
        inicio = fim + 1
    return resultado


def figura_svg(
    grupos: list[dict[str, object]], correlacoes: list[dict[str, object]], agrupado: bool = False
) -> str:
    """Cria uma grade 3 × 3 de dispersões sem depender de bibliotecas externas."""
    largura, altura = 1320, 980
    margem_x, margem_y = 80, 86
    painel_largura, painel_altura = 390, 270
    cores = {"BR16": "#2b6cb0", "CD202": "#c05621", "EMB48": "#2f855a"}
    medidas = [("chl_a_media", "Chl a"), ("chl_b_media", "Chl b"), ("chl_total_media", "Chl total")]
    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1a202c}.title{font-size:24px;font-weight:bold}.panel{font-size:16px;font-weight:bold}.axis{font-size:13px}.note{font-size:12px;fill:#4a5568}</style>',
        f'<text x="660" y="38" text-anchor="middle" class="title">R680 × clorofilas — {"genótipos agrupados" if agrupado else "médias de genótipo × condição"}</text>',
        f'<text x="80" y="63" class="note">Espectros da manhã; {2 if agrupado else 6} grupos por painel. {"Azul = irrigado; laranja = não irrigado." if agrupado else "Círculo = irrigado; quadrado = não irrigado."}</text>',
    ]
    for coluna, (_, rotulo) in enumerate(medidas):
        x = margem_x + coluna * painel_largura + painel_largura / 2
        partes.append(f'<text x="{x:.0f}" y="84" text-anchor="middle" class="panel">{rotulo}</text>')

    for linha, dia in enumerate(DIAS):
        dados_dia = [item for item in grupos if item["dia"] == dia]
        for coluna, (campo, rotulo) in enumerate(medidas):
            esquerda = margem_x + coluna * painel_largura
            topo = margem_y + 20 + linha * painel_altura
            direita, base = esquerda + 320, topo + 190
            xval = [float(item["r680_medio"]) for item in dados_dia]
            yval = [float(item[campo]) for item in dados_dia]
            dx, dy = max(xval) - min(xval), max(yval) - min(yval)
            xmin, xmax = min(xval) - max(dx * .12, .001), max(xval) + max(dx * .12, .001)
            ymin, ymax = min(yval) - max(dy * .12, .5), max(yval) + max(dy * .12, .5)
            xp = lambda v: esquerda + (v - xmin) / (xmax - xmin) * (direita - esquerda)
            yp = lambda v: base - (v - ymin) / (ymax - ymin) * (base - topo)
            partes.extend([
                f'<rect x="{esquerda}" y="{topo}" width="{direita-esquerda}" height="{base-topo}" fill="#fff" stroke="#a0aec0"/>',
                f'<text x="{esquerda}" y="{topo-8}" class="panel">{dia} ({DIAS[dia]})</text>',
                f'<text x="{(esquerda+direita)/2:.1f}" y="{base+32}" text-anchor="middle" class="axis">Reflectância em 680 nm</text>',
                f'<text transform="translate({esquerda-46},{(topo+base)/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">{rotulo}</text>',
                f'<text x="{esquerda}" y="{base+14}" class="note">{xmin:.3f}</text>',
                f'<text x="{direita}" y="{base+14}" text-anchor="end" class="note">{xmax:.3f}</text>',
                f'<text x="{esquerda-7}" y="{base}" text-anchor="end" class="note">{ymin:.1f}</text>',
                f'<text x="{esquerda-7}" y="{topo+4}" text-anchor="end" class="note">{ymax:.1f}</text>',
            ])
            # Reta de mínimos quadrados, apenas como guia visual.
            media_x, media_y = mean(xval), mean(yval)
            den = sum((v - media_x) ** 2 for v in xval)
            inclinacao = sum((a - media_x) * (b - media_y) for a, b in zip(xval, yval)) / den
            intercepto = media_y - inclinacao * media_x
            y_reta_min = max(ymin, min(ymax, inclinacao * xmin + intercepto))
            y_reta_max = max(ymin, min(ymax, inclinacao * xmax + intercepto))
            partes.append(f'<line x1="{xp(xmin):.1f}" y1="{yp(y_reta_min):.1f}" x2="{xp(xmax):.1f}" y2="{yp(y_reta_max):.1f}" stroke="#718096" stroke-width="1.5" stroke-dasharray="5 4"/>')
            resultado = next(item for item in correlacoes if item["dia"] == dia and item["variavel_fisiologica"] == rotulo.upper().replace(" ", "_"))
            partes.append(f'<text x="{direita}" y="{topo+17}" text-anchor="end" class="note">r = {float(resultado["correlacao_pearson_r"]):.2f}; R² = {float(resultado["r_quadrado"]):.2f}</text>')
            for item in dados_dia:
                xcoord, ycoord = xp(float(item["r680_medio"])), yp(float(item[campo]))
                cor = ("#2b6cb0" if item["condicao"] == "IRR" else "#c05621") if agrupado else cores[str(item["genotipo"])]
                if agrupado or item["condicao"] == "IRR":
                    partes.append(f'<circle cx="{xcoord:.1f}" cy="{ycoord:.1f}" r="6" fill="{cor}" stroke="white" stroke-width="1.5"/>')
                else:
                    partes.append(f'<rect x="{xcoord-5:.1f}" y="{ycoord-5:.1f}" width="10" height="10" fill="{cor}" stroke="white" stroke-width="1.5"/>')
    partes.append('</svg>')
    return "\n".join(partes)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    planilha = ler_fisiologico()
    cabecalho, dados_fisio = planilha[0], planilha[1:]

    fisiologico = defaultdict(list)
    for dia, data in DIAS.items():
        colunas = {
            nome: cabecalho.index(f"{nome} ({data})")
            for nome in ("CHL A", "CHL B", "CHL TOTAL")
        }
        for linha in dados_fisio:
            grupo = (linha[0], linha[1])
            for nome, indice in colunas.items():
                fisiologico[(dia, *grupo, nome)].append(numero(linha[indice]))

    espectral = defaultdict(list)
    with ESPECTRAL.open(encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo, delimiter=";"):
            dia = linha["data_coleta"][:3]
            if dia not in DIAS or linha["turno"] != "manha":
                continue
            condicao = {"IRRIG": "IRR", "NIRRIG": "NIR"}[linha["condicao"]]
            espectral[(dia, linha["genotipo"], condicao)].append(numero(linha["680"]))

    grupos = []
    for dia in DIAS:
        for genotipo in ("BR16", "CD202", "EMB48"):
            for condicao in ("IRR", "NIR"):
                r680 = espectral[(dia, genotipo, condicao)]
                if not r680:
                    raise ValueError(f"Sem espectros para {dia}/{genotipo}/{condicao}")
                grupos.append(
                    {
                        "dia": dia,
                        "data_fisiologica": DIAS[dia],
                        "genotipo": genotipo,
                        "condicao": condicao,
                        "n_espectros": len(r680),
                        "r680_medio": mean(r680),
                        **{
                            f"{nome.lower().replace(' ', '_')}_media": mean(
                                fisiologico[(dia, genotipo, condicao, nome)]
                            )
                            for nome in ("CHL A", "CHL B", "CHL TOTAL")
                        },
                    }
                )

    with (SAIDA / "r680_clorofila_medias_grupo.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=grupos[0].keys(), delimiter=";")
        escritor.writeheader()
        escritor.writerows(grupos)

    correlacoes = []
    for dia in DIAS:
        linhas_dia = [linha for linha in grupos if linha["dia"] == dia]
        x = [linha["r680_medio"] for linha in linhas_dia]
        for variavel in ("chl_a_media", "chl_b_media", "chl_total_media"):
            y = [linha[variavel] for linha in linhas_dia]
            r_pearson = correlacao_pearson(x, y)
            correlacoes.append(
                {
                    "dia": dia,
                    "data_fisiologica": DIAS[dia],
                    "variavel_fisiologica": variavel.replace("_media", "").upper(),
                    "n_grupos": len(linhas_dia),
                    "correlacao_pearson_r": r_pearson,
                    "r_quadrado": r_pearson**2,
                    "correlacao_spearman_rho": correlacao_pearson(postos(x), postos(y)),
                }
            )

    with (SAIDA / "r680_clorofila_correlacoes.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=correlacoes[0].keys(), delimiter=";")
        escritor.writeheader()
        escritor.writerows(correlacoes)

    (SAIDA / "r680_clorofila.svg").write_text(figura_svg(grupos, correlacoes), encoding="utf-8")

    # Agrega os três genótipos, mantendo as duas condições em cada dia.
    grupos_agrupados = []
    for dia in DIAS:
        for condicao in ("IRR", "NIR"):
            linhas = [item for item in grupos if item["dia"] == dia and item["condicao"] == condicao]
            grupos_agrupados.append(
                {
                    "dia": dia,
                    "data_fisiologica": DIAS[dia],
                    "condicao": condicao,
                    "n_genotipos": len(linhas),
                    "n_espectros": sum(int(item["n_espectros"]) for item in linhas),
                    **{campo: mean(float(item[campo]) for item in linhas) for campo in ("r680_medio", "chl_a_media", "chl_b_media", "chl_total_media")},
                }
            )
    with (SAIDA / "r680_clorofila_genotipos_agrupados_medias.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=grupos_agrupados[0].keys(), delimiter=";")
        escritor.writeheader()
        escritor.writerows(grupos_agrupados)

    correlacoes_agrupadas = []
    for dia in DIAS:
        linhas = [item for item in grupos_agrupados if item["dia"] == dia]
        x = [float(item["r680_medio"]) for item in linhas]
        for campo in ("chl_a_media", "chl_b_media", "chl_total_media"):
            r_pearson = correlacao_pearson(x, [float(item[campo]) for item in linhas])
            correlacoes_agrupadas.append({"dia": dia, "data_fisiologica": DIAS[dia], "variavel_fisiologica": campo.replace("_media", "").upper(), "n_grupos": 2, "correlacao_pearson_r": r_pearson, "r_quadrado": r_pearson**2})
    with (SAIDA / "r680_clorofila_genotipos_agrupados_correlacoes.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=correlacoes_agrupadas[0].keys(), delimiter=";")
        escritor.writeheader()
        escritor.writerows(correlacoes_agrupadas)
    (SAIDA / "r680_clorofila_genotipos_agrupados.svg").write_text(figura_svg(grupos_agrupados, correlacoes_agrupadas, agrupado=True), encoding="utf-8")


if __name__ == "__main__":
    main()
