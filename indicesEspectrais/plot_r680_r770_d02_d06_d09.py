#!/usr/bin/env python3
"""Plota R680 e R770 brutas para IRRIG × NIRRIG em D02, D06 e D09.

Gera uma figura independente para cada comprimento de onda, sem calcular
índices espectrais. Cada coluna é um dia, e cada par de barras dentro do
painel compara IRRIG e NIRRIG para um genótipo. Os pontos são médias das
leituras da manhã e os traços verticais mostram IC de 95%.

Uso:
    python indicesEspectrais/plot_r680_r770_d02_d06_d09.py

Saídas:
    indicesEspectrais/r680_d02_d06_d09_irr_x_nirr.svg
    indicesEspectrais/r680_d02_d06_d09_irr_x_nirr.png
    indicesEspectrais/r770_d02_d06_d09_irr_x_nirr.svg
    indicesEspectrais/r770_d02_d06_d09_irr_x_nirr.png
    indicesEspectrais/r680_r770_d02_d06_d09_resumo.csv
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = ROOT.parent / "dataset" / "Unificada13052026_Limpa.csv"
DIAS = ("D02", "D06", "D09")
GENOTIPOS = ("BR16", "CD202", "EMB48")
CONDICOES = ("IRRIG", "NIRRIG")
BANDAS = ("680", "770")
CORES = {"IRRIG": "#2a78d6", "NIRRIG": "#eb6834"}

# Valores t bilaterais para 95% de confiança; os grupos da base têm n=32 ou 36.
T_CRITICO_95 = {31: 2.0395, 35: 2.0301}


def numero(valor: str) -> float:
    """Converte a notação decimal com vírgula usada no arquivo de entrada."""
    return float(valor.strip().replace(",", "."))


def carregar() -> dict[tuple[str, str, str, str], list[float]]:
    """Lê as reflectâncias brutas da manhã nos dias e grupos selecionados."""
    grupos: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    with ARQUIVO_ENTRADA.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo, delimiter=";")
        for linha in leitor:
            dia = linha["data_coleta"][:3]
            if (
                dia not in DIAS
                or linha["turno"].strip().lower() != "manha"
                or linha["genotipo"] not in GENOTIPOS
                or linha["condicao"] not in CONDICOES
            ):
                continue
            for banda in BANDAS:
                try:
                    grupos[(banda, dia, linha["genotipo"], linha["condicao"])].append(
                        numero(linha[banda])
                    )
                except (KeyError, ValueError):
                    continue
    faltantes = [
        (banda, dia, genotipo, condicao)
        for banda in BANDAS
        for dia in DIAS
        for genotipo in GENOTIPOS
        for condicao in CONDICOES
        if not grupos[(banda, dia, genotipo, condicao)]
    ]
    if faltantes:
        raise ValueError(f"Grupos sem leituras: {faltantes}")
    return grupos


def resumir(grupos: dict[tuple[str, str, str, str], list[float]]) -> list[dict[str, object]]:
    """Calcula média e IC95% diretamente para cada banda, dia e tratamento."""
    resumo = []
    for banda in BANDAS:
        for dia in DIAS:
            for genotipo in GENOTIPOS:
                for condicao in CONDICOES:
                    valores = grupos[(banda, dia, genotipo, condicao)]
                    n = len(valores)
                    media = mean(valores)
                    erro = stdev(valores) / math.sqrt(n) if n > 1 else 0.0
                    t = T_CRITICO_95.get(n - 1, 1.96)
                    margem = t * erro
                    resumo.append(
                        {
                            "banda_nm": int(banda), "dia": dia,
                            "genotipo": genotipo, "condicao": condicao,
                            "n": n, "media_reflectancia": media,
                            "desvio_padrao": stdev(valores) if n > 1 else 0.0,
                            "ic95_inferior": media - margem,
                            "ic95_superior": media + margem,
                        }
                    )
    return resumo


def escala(valor: float, minimo: float, maximo: float, y_topo: float, altura: float) -> float:
    return y_topo + altura * (maximo - valor) / (maximo - minimo)


def texto(x: float, y: float, conteudo: str, tamanho: float = 13, ancora: str = "middle",
          peso: str = "normal", cor: str = "#1a1a1a") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{ancora}" '
            f'font-family="Arial, sans-serif" font-size="{tamanho}" font-weight="{peso}" '
            f'fill="{cor}">{escape(conteudo)}</text>')


def plotar(banda: str, resumo: list[dict[str, object]]) -> Path:
    """Desenha um SVG com três painéis, um por dia, para uma banda bruta."""
    dados = [r for r in resumo if r["banda_nm"] == int(banda)]
    minimo = min(float(r["ic95_inferior"]) for r in dados)
    maximo = max(float(r["ic95_superior"]) for r in dados)
    folga = max((maximo - minimo) * 0.12, 0.006)
    minimo = max(0.0, minimo - folga)
    maximo += folga

    largura, altura_total = 1500, 670
    margem_esq, margem_dir, topo, base = 105, 40, 135, 115
    largura_painel = (largura - margem_esq - margem_dir) / 3
    altura_painel = altura_total - topo - base
    linhas = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura_total}" viewBox="0 0 {largura} {altura_total}">',
        '<rect width="100%" height="100%" fill="white"/>',
        texto(largura / 2, 42, f"Reflectância bruta R{banda}: IRRIG × NIRRIG", 25, peso="bold"),
        texto(largura / 2, 72, "Leituras do turno da manhã; pontos = média e traços = IC de 95%", 15, cor="#4f5964"),
    ]
    # Eixo y compartilhado, com cinco marcas internas e uma no limite superior.
    for i in range(6):
        valor = minimo + (maximo - minimo) * i / 5
        y = escala(valor, minimo, maximo, topo, altura_painel)
        linhas.append(f'<line x1="{margem_esq}" y1="{y:.1f}" x2="{largura - margem_dir}" y2="{y:.1f}" stroke="#d7dde3" stroke-width="1"/>')
        linhas.append(texto(margem_esq - 12, y + 5, f"{valor:.3f}", 12, ancora="end", cor="#4f5964"))
    linhas.append(f'<line x1="{margem_esq}" y1="{topo}" x2="{margem_esq}" y2="{topo + altura_painel}" stroke="#45505c" stroke-width="1.4"/>')
    linhas.append(texto(25, topo + altura_painel / 2, "Reflectância", 15, ancora="middle", peso="bold"))

    for indice_dia, dia in enumerate(DIAS):
        x0 = margem_esq + indice_dia * largura_painel
        x1 = x0 + largura_painel
        if indice_dia:
            linhas.append(f'<line x1="{x0:.1f}" y1="{topo}" x2="{x0:.1f}" y2="{topo + altura_painel}" stroke="#aab3bd" stroke-width="1" stroke-dasharray="4 5"/>')
        linhas.append(texto((x0 + x1) / 2, 115, dia, 19, peso="bold"))
        for indice_genotipo, genotipo in enumerate(GENOTIPOS):
            centro = x0 + largura_painel * (indice_genotipo + 0.5) / len(GENOTIPOS)
            for deslocamento, condicao in ((-37, "IRRIG"), (37, "NIRRIG")):
                registro = next(r for r in dados if r["dia"] == dia and r["genotipo"] == genotipo and r["condicao"] == condicao)
                y_media = escala(float(registro["media_reflectancia"]), minimo, maximo, topo, altura_painel)
                y_inf = escala(float(registro["ic95_inferior"]), minimo, maximo, topo, altura_painel)
                y_sup = escala(float(registro["ic95_superior"]), minimo, maximo, topo, altura_painel)
                x = centro + deslocamento
                linhas.append(f'<line x1="{x:.1f}" y1="{y_inf:.1f}" x2="{x:.1f}" y2="{y_sup:.1f}" stroke="#29323b" stroke-width="2"/>')
                linhas.append(f'<line x1="{x - 8:.1f}" y1="{y_inf:.1f}" x2="{x + 8:.1f}" y2="{y_inf:.1f}" stroke="#29323b" stroke-width="2"/>')
                linhas.append(f'<line x1="{x - 8:.1f}" y1="{y_sup:.1f}" x2="{x + 8:.1f}" y2="{y_sup:.1f}" stroke="#29323b" stroke-width="2"/>')
                linhas.append(f'<circle cx="{x:.1f}" cy="{y_media:.1f}" r="9" fill="{CORES[condicao]}" stroke="white" stroke-width="2"/>')
            linhas.append(texto(centro, topo + altura_painel + 28, genotipo, 14, peso="bold"))
    # Legenda.
    legenda_x = largura / 2 - 125
    for i, condicao in enumerate(CONDICOES):
        x = legenda_x + i * 190
        linhas.append(f'<circle cx="{x + 10}" cy="{altura_total - 48}" r="8" fill="{CORES[condicao]}"/>')
        linhas.append(texto(x + 28, altura_total - 43, condicao, 14, ancora="start"))
    linhas.append('</svg>')
    destino = ROOT / f"r{banda}_d02_d06_d09_irr_x_nirr.svg"
    destino.write_text("\n".join(linhas), encoding="utf-8")
    return destino


def converter_png(svg: Path) -> None:
    """Cria um PNG de alta resolução quando ImageMagick estiver disponível."""
    conversor = shutil.which("magick") or shutil.which("convert")
    if conversor:
        subprocess.run([conversor, "-density", "220", str(svg), str(svg.with_suffix(".png"))], check=True)


def salvar_resumo(resumo: list[dict[str, object]]) -> Path:
    destino = ROOT / "r680_r770_d02_d06_d09_resumo.csv"
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        campos = list(resumo[0])
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for linha in resumo:
            escritor.writerow(linha)
    return destino


def main() -> None:
    grupos = carregar()
    resumo = resumir(grupos)
    saidas = [plotar(banda, resumo) for banda in BANDAS]
    for svg in saidas:
        converter_png(svg)
    csv_saida = salvar_resumo(resumo)
    print("Gráficos salvos em:")
    for svg in saidas:
        print(f"  {svg}")
        print(f"  {svg.with_suffix('.png')}")
    print(f"Resumo salvo em: {csv_saida}")


if __name__ == "__main__":
    main()
