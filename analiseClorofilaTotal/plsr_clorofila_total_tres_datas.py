#!/usr/bin/env python3
"""Executa o PLSR de CHL TOTAL nas suas sete coletas.

Reutiliza o fluxo de pareamento, reducao Spearman, VIP e validacao por bloco
do modulo de Yield; somente o alvo fisiologico muda.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analiseYield"))
import plsr_yield_tres_datas as base  # noqa: E402

base.OUT = ROOT / "analiseClorofilaTotal" / "resultados"
base.MAPA = {
    "23_02": ("CHL TOTAL (23/02)", "D02M", "23/02"),
    "24_02": ("CHL TOTAL (24/02)", "D03M", "24/02"),
    "25_02": ("CHL TOTAL (25.02)", "D04M", "25/02"),
    "26_02": ("CHL TOTAL (26/02)", "D05M", "26/02"),
    "27_02": ("CHL TOTAL (27/02)", "D06M", "27/02"),
    "02_03": ("CHL TOTAL (02/03)", "D09M", "02/03"),
    "03_03": ("CHL TOTAL (03/03)", "D10M", "03/03"),
}


def main():
    base.OUT.mkdir(parents=True, exist_ok=True)
    fis, esp, bandas = base.preparar()
    resultados = [base.analisar(nome, *args, fis, esp, bandas) for nome, args in base.MAPA.items()]
    for nome in base.MAPA:
        arquivo = base.OUT / f"{nome}_bandas_vip_ge_1.csv"
        ranking = base.pd.read_csv(arquivo, sep=";").rename(columns={
            "data_yield": "data_clorofila_total",
            "rho_spearman_yield": "rho_spearman_clorofila_total",
        })
        ranking.to_csv(arquivo, sep=";", index=False)
    resumo = base.pd.DataFrame(resultados)
    resumo.to_csv(base.OUT / "resumo_plsr_clorofila_total.csv", sep=";", index=False)
    linhas = [
        "# PLSR — Clorofila Total nas sete coletas", "",
        "Pareamento: média de leituras espectrais matinais por bloco × genótipo × condição; D02M–D06M=23–27/02, D09M=02/03 e D10M=03/03.", "",
        "Spearman |ρ| ≥ 0,80 entre bandas contíguas (janela <10 nm); uma representante por grupo, escolhida pelo maior |ρ| com Clorofila Total. O PLSR é escalonado, com componentes escolhidos pelo menor RMSE em CV deixando um bloco de fora. VIP ≥1,0 do ajuste inicial é retido e o modelo final é reajustado.", "",
        "`R² CV-bloco` é diagnóstico: a seleção Spearman/VIP ocorreu antes das dobras e pode tornar a estimativa otimista.", "",
        "| Clorofila Total | Coleta espectral | n | Rep. Spearman | VIP ≥1 | Componentes | R² ajuste | R² CV-bloco | RMSE CV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in resultados:
        linhas.append(f"| {r['data_yield']} | {r['coleta_espectral']} | {r['n']} | {r['representantes_spearman']} | {r['bandas_vip_ge_1']} | {r['componentes']} | {r['r2_ajuste']:.3f} | {r['r2_cv_bloco']:.3f} | {r['rmse_cv_bloco']:.4f} |")
    (base.OUT / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
