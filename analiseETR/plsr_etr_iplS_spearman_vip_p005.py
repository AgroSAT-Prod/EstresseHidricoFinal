#!/usr/bin/env python3
"""Executa a análise iPLS/PLSR de ETR com Spearman p < 0,05."""
from pathlib import Path

import plsr_etr_iplS_spearman_vip as base

base.P_LIMIAR = 0.05
base.OUT = Path(__file__).resolve().parent / "resultados_p005"


if __name__ == "__main__":
    base.main()
