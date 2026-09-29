#!/usr/bin/env python3
"""PCA espectral por dia e comparação entre loadings e ΔRλ.

Usa a reflectância original da base limpa, centrada por comprimento de onda
(sem padronização por desvio-padrão). A PCA é calculada separadamente para
D04M, D05M, D06M e D09M. O sinal de cada componente é orientado para que a
média dos escores de IRRIG menos NIRRIG seja positiva; o sinal não altera
nem a importância absoluta do loading nem a variância explicada.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "dataset" / "Unificada13052026_Limpa.csv"
SAIDA = Path(__file__).resolve().parent / "dataset_gerado"
DIAS = ("D04M", "D05M", "D06M", "D09M")
TOP_N = 20


def produto(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalizar(vetor: list[float]) -> list[float]:
    norma = math.sqrt(produto(vetor, vetor))
    if norma == 0:
        raise ValueError("Vetor nulo durante o cálculo da PCA.")
    return [valor / norma for valor in vetor]


def covariancia_vezes_vetor(matriz: list[list[float]], vetor: list[float]) -> list[float]:
    """Calcula X'Xv sem construir a matriz de covariância (2.151 × 2.151)."""
    resultado = [0.0] * len(vetor)
    for linha in matriz:
        escore = produto(linha, vetor)
        for indice, valor in enumerate(linha):
            resultado[indice] += valor * escore
    return resultado


def componente_principal(matriz: list[list[float]], semente: int) -> tuple[list[float], list[float]]:
    vetor = normalizar([math.sin((indice + 1) * (semente + 1)) for indice in range(len(matriz[0]))])
    for _ in range(2_000):
        novo = normalizar(covariancia_vezes_vetor(matriz, vetor))
        if abs(produto(novo, vetor)) > 1 - 1e-12:
            vetor = novo
            break
        vetor = novo
    escores = [produto(linha, vetor) for linha in matriz]
    return vetor, escores


def pca_duas_componentes(matriz: list[list[float]], condicoes: list[str]) -> tuple[list[float], list[float], float, float]:
    residual = [linha[:] for linha in matriz]
    componentes, variancias = [], []
    variancia_total = sum(produto(linha, linha) for linha in matriz) / (len(matriz) - 1)
    for numero in range(2):
        loading, escores = componente_principal(residual, numero)
        diferenca_escore = (
            sum(escore for escore, condicao in zip(escores, condicoes) if condicao == "IRRIG") / condicoes.count("IRRIG")
            - sum(escore for escore, condicao in zip(escores, condicoes) if condicao == "NIRRIG") / condicoes.count("NIRRIG")
        )
        if diferenca_escore < 0:
            loading, escores = [-valor for valor in loading], [-valor for valor in escores]
        autovalor = sum(escore * escore for escore in escores) / (len(matriz) - 1)
        variancias.append(autovalor / variancia_total)
        componentes.append(loading)
        for linha, escore in zip(residual, escores):
            for indice, valor in enumerate(loading):
                linha[indice] -= escore * valor
    return componentes[0], componentes[1], variancias[0], variancias[1]


def limites(valores: list[float]) -> tuple[float, float]:
    minimo, maximo = min(valores), max(valores)
    amplitude = maximo - minimo
    folga = amplitude * 0.08 if amplitude else 1.0
    return minimo - folga, maximo + folga


def polilinha(x: list[int], y: list[float], esquerda: float, topo: float, largura: float, altura: float) -> str:
    xmin, xmax = min(x), max(x)
    ymin, ymax = limites(y)
    pontos = []
    for xv, yv in zip(x, y):
        px = esquerda + (xv - xmin) / (xmax - xmin) * largura
        py = topo + altura - (yv - ymin) / (ymax - ymin) * altura
        pontos.append(f"{px:.1f},{py:.1f}")
    zero = topo + altura - (0 - ymin) / (ymax - ymin) * altura
    eixo_zero = f'<line x1="{esquerda}" y1="{zero:.1f}" x2="{esquerda+largura}" y2="{zero:.1f}" stroke="#a0aec0" stroke-width="0.8"/>' if topo <= zero <= topo + altura else ""
    return f'{eixo_zero}<polyline points="{" ".join(pontos)}" fill="none" stroke="#2b6cb0" stroke-width="1"/>'


def figura_svg(dia: str, bandas: list[int], pc1: list[float], pc2: list[float], delta: list[float], var1: float, var2: float) -> str:
    largura, altura = 1400, 800
    paineis = [("Loading PC1", pc1, f"PC1: {var1:.1%} da variância"), ("Loading PC2", pc2, f"PC2: {var2:.1%} da variância"), ("ΔRλ (IRRIG − NIRRIG)", delta, "Diferença entre médias de reflectância")]
    partes = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#1a202c}.title{font-size:25px;font-weight:bold}.panel{font-size:17px;font-weight:bold}.note{font-size:13px;fill:#4a5568}</style>', f'<text x="700" y="38" text-anchor="middle" class="title">{dia} — loadings de PCA e ΔRλ</text>', '<text x="80" y="62" class="note">PCA em reflectância centrada por banda; sinais de PC orientados para IRRIG − NIRRIG positivo.</text>']
    for indice, (titulo, valores, subtitulo) in enumerate(paineis):
        topo = 88 + indice * 225
        partes.extend([f'<text x="80" y="{topo-10}" class="panel">{titulo}</text>', f'<text x="300" y="{topo-10}" class="note">{subtitulo}</text>', f'<rect x="80" y="{topo}" width="1230" height="160" fill="white" stroke="#a0aec0"/>', polilinha(bandas, valores, 80, topo, 1230, 160), f'<text x="80" y="{topo+182}" class="note">350 nm</text>', f'<text x="1310" y="{topo+182}" text-anchor="end" class="note">2500 nm</text>'])
    partes.append('</svg>')
    return "\n".join(partes)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    dados = defaultdict(list)
    with ENTRADA.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo, delimiter=";")
        bandas = [int(coluna) for coluna in leitor.fieldnames if coluna.isdigit()]
        colunas = [str(banda) for banda in bandas]
        for linha in leitor:
            if linha["data_coleta"] in DIAS:
                dados[linha["data_coleta"]].append((linha["condicao"], [float(linha[coluna].replace(",", ".")) for coluna in colunas]))

    resumo, top_bandas = [], []
    for dia in DIAS:
        observacoes = dados[dia]
        condicoes = [condicao for condicao, _ in observacoes]
        matriz_original = [valores for _, valores in observacoes]
        medias = [sum(linha[indice] for linha in matriz_original) / len(matriz_original) for indice in range(len(bandas))]
        matriz_centrada = [[valor - medias[indice] for indice, valor in enumerate(linha)] for linha in matriz_original]
        pc1, pc2, var1, var2 = pca_duas_componentes(matriz_centrada, condicoes)
        irr = [linha for condicao, linha in observacoes if condicao == "IRRIG"]
        nir = [linha for condicao, linha in observacoes if condicao == "NIRRIG"]
        delta = [sum(linha[indice] for linha in irr) / len(irr) - sum(linha[indice] for linha in nir) / len(nir) for indice in range(len(bandas))]
        ordem_pc1 = sorted(range(len(bandas)), key=lambda i: abs(pc1[i]), reverse=True)
        ordem_pc2 = sorted(range(len(bandas)), key=lambda i: abs(pc2[i]), reverse=True)
        ordem_delta = sorted(range(len(bandas)), key=lambda i: abs(delta[i]), reverse=True)
        rank_pc1 = {indice: posicao + 1 for posicao, indice in enumerate(ordem_pc1)}
        rank_pc2 = {indice: posicao + 1 for posicao, indice in enumerate(ordem_pc2)}
        rank_delta = {indice: posicao + 1 for posicao, indice in enumerate(ordem_delta)}
        tabela = []
        top_delta_nm = [bandas[indice] for indice in ordem_delta[:TOP_N]]
        for indice, banda in enumerate(bandas):
            tabela.append({"dia": dia, "comprimento_onda_nm": banda, "loading_pc1": pc1[indice], "loading_pc2": pc2[indice], "delta_r_irr_minus_nir": delta[indice], "abs_loading_pc1": abs(pc1[indice]), "abs_loading_pc2": abs(pc2[indice]), "abs_delta_r": abs(delta[indice]), "rank_abs_loading_pc1": rank_pc1[indice], "rank_abs_loading_pc2": rank_pc2[indice], "rank_abs_delta_r": rank_delta[indice], "distancia_nm_ao_top20_delta": min(abs(banda - alvo) for alvo in top_delta_nm)})
        with (SAIDA / f"{dia}_loadings_pc1_pc2_delta_r.csv").open("w", encoding="utf-8", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=tabela[0].keys(), delimiter=";")
            escritor.writeheader(); escritor.writerows(tabela)
        for componente, ordem in (("PC1", ordem_pc1), ("PC2", ordem_pc2), ("DELTA_R", ordem_delta)):
            for posicao, indice in enumerate(ordem[:TOP_N], 1):
                top_bandas.append({"dia": dia, "metrica": componente, "posicao": posicao, "comprimento_onda_nm": bandas[indice], "loading_pc1": pc1[indice], "loading_pc2": pc2[indice], "delta_r_irr_minus_nir": delta[indice], "distancia_nm_ao_top20_delta": min(abs(bandas[indice] - alvo) for alvo in top_delta_nm)})
        resumo.append({"dia": dia, "n_espectros": len(observacoes), "n_irr": len(irr), "n_nir": len(nir), "variancia_explicada_pc1": var1, "variancia_explicada_pc2": var2, "variancia_explicada_pc1_pc2": var1 + var2, "n_top20_pc1_a_10nm_do_top20_delta": sum(any(abs(bandas[indice] - alvo) <= 10 for alvo in top_delta_nm) for indice in ordem_pc1[:TOP_N]), "n_top20_pc2_a_10nm_do_top20_delta": sum(any(abs(bandas[indice] - alvo) <= 10 for alvo in top_delta_nm) for indice in ordem_pc2[:TOP_N])})
        (SAIDA / f"{dia}_loadings_pc1_pc2_delta_r.svg").write_text(figura_svg(dia, bandas, pc1, pc2, delta, var1, var2), encoding="utf-8")
    with (SAIDA / "resumo_pca_delta_r.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=resumo[0].keys(), delimiter=";")
        escritor.writeheader(); escritor.writerows(resumo)
    with (SAIDA / "top20_loadings_e_delta_r.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=top_bandas[0].keys(), delimiter=";")
        escritor.writeheader(); escritor.writerows(top_bandas)


if __name__ == "__main__":
    main()
