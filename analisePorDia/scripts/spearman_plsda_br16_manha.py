#!/usr/bin/env python3
"""Compatibilidade para o antigo experimento Spearman 0,80 por genotipo.

O motor vigente esta em ``selecaoVariaveis/experimento_spearman_por_dia.py``.
Este comando preserva o argumento ``--genotipo`` do script anterior e executa
somente o limiar 0,80, sem alterar os resultados legados organizados por dia.
Esse limiar forma grupos com |rho| > 0,80 e separa o Top com |rho| < 0,80.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "selecaoVariaveis"))

from experimento_spearman_por_dia import executar_e_salvar  # noqa: E402


SAIDA_COMPATIBILIDADE = (
    ROOT / "analisePorDia" / "resultados" / "dataset_gerado"
    / "experimento_spearman_compatibilidade"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibilidade: agrupamento Spearman >0,80 e Top com "
            "correlacao maxima <0,80 por dia e genotipo."
        )
    )
    parser.add_argument(
        "--genotipo", default="BR16", choices=("BR16", "CD202", "EMB48")
    )
    parser.add_argument(
        "--saida", type=Path, default=SAIDA_COMPATIBILIDADE,
        help="Pasta das seis tabelas consolidadas da execucao de compatibilidade.",
    )
    args = parser.parse_args()
    print(
        "AVISO: este e o entrypoint de compatibilidade. "
        "O experimento multilimiar fica em selecaoVariaveis/.\n"
    )
    executar_e_salvar(
        genotipos=(args.genotipo,),
        correlacoes_maximas=(0.80,),
        saida=args.saida,
    )


if __name__ == "__main__":
    main()
