#!/usr/bin/env python3
"""Executa para CRA a seleção PLSR/VIP independente por data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analiseClorofilaTotal"))
import plsr_chltotal_vip_por_dia as pipeline  # noqa: E402

pipeline.ALVO_NOME = "CRA"
pipeline.ALVO_CHAVE = "cra"
pipeline.OUT = ROOT / "analiseCRA" / "resultados" / "plsr_cra_vip_por_dia_top5"
pipeline.base.MAPA = {
    "24_02": ("24/02", "CRA (24/02)", "D03M"),
    "02_03": ("02/03", "CRA (02/03)", "D09M"),
    "03_03": ("03/03", "CRA (03/03)", "D10M"),
}

if __name__ == "__main__":
    pipeline.main()
