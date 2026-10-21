#!/usr/bin/env python3
"""Executa para evapotranspiração (E) a pipeline PLSR global usada em CHL Total."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analiseClorofilaTotal"))
import plsr_chltotal_global_top5_por_dia as pipeline  # noqa: E402

pipeline.ALVO_NOME = "Evapotranspiração (E)"
pipeline.ALVO_CHAVE = "e"
pipeline.OUT = ROOT / "analiseETR" / "resultados" / "plsr_e_global_top5_por_dia"
pipeline.MAPA = {
    "23_02": ("23/02", "E (23/02)", "D02M"),
    "24_02": ("24/02", "E (24/02)", "D03M"),
    "25_02": ("25/02", "E (25/02)", "D04M"),
    "26_02": ("26/02", "E (26/02)", "D05M"),
    "27_02": ("27/02", "E (27/02)", "D06M"),
    "02_03": ("02/03", "E (02/03)", "D09M"),
    "03_03": ("03/03", "E (03/03)", "D10M"),
}

if __name__ == "__main__":
    pipeline.main()
