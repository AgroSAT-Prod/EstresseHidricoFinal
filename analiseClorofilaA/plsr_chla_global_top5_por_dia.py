#!/usr/bin/env python3
"""Executa para CHL A a pipeline PLSR global padrão."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analiseClorofilaTotal"))
import plsr_chltotal_global_top5_por_dia as pipeline  # noqa: E402

pipeline.ALVO_NOME = "CHL A"
pipeline.ALVO_CHAVE = "chl_a"
pipeline.OUT = ROOT / "analiseClorofilaA" / "resultados" / "plsr_chla_global_top5_por_dia"
pipeline.MAPA = {
    "23_02": ("23/02", "CHL A (23/02)", "D02M"),
    "24_02": ("24/02", "CHL A (24/02)", "D03M"),
    "25_02": ("25/02", "CHL A (25/02)", "D04M"),
    "26_02": ("26/02", "CHL A (26/02)", "D05M"),
    "27_02": ("27/02", "CHL A (27/02)", "D06M"),
    "02_03": ("02/03", "CHL A (02/03)", "D09M"),
    "03_03": ("03/03", "CHL A (03/03)", "D10M"),
}

if __name__ == "__main__":
    pipeline.main()
