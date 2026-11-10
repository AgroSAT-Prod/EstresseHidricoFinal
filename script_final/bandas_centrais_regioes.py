"""Centros das feições na curva média global da classificação."""
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
contexto = runpy.run_path(str(PASTA / "fwhm_regioes_classificacao.py"))
base = contexto["base"]
linhas = []
for numero, (inicio, fim) in enumerate(contexto["regioes"], 1):
    x = np.arange(inicio, fim + 1)
    y = base[[str(w) for w in x]].mean().to_numpy(dtype=float)
    residuo = y - np.interp(x, [inicio, fim], [y[0], y[-1]])
    indice = int(np.argmax(np.abs(residuo)))
    linhas.append(dict(regiao=numero, extremo_esquerdo_nm=inicio, extremo_direito_nm=fim,
                       banda_central_nm=int(x[indice]), tipo="pico" if residuo[indice] > 0 else "vale", n=len(base)))
resultado = pd.DataFrame(linhas)
resultado.to_csv(PASTA / "outputs/classificacao/bandas_centrais_regioes.csv", sep=";", index=False)
print(resultado.to_string(index=False))
