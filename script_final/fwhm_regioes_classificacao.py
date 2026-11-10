"""FWHM do principal desvio da curva média em relação à linha entre extremos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
notebook = json.loads((ROOT / "script_final/Pipeline_Final_Classificacao.ipynb").read_text(encoding="utf-8"))
ambiente = {}
for marcador in ("def localizar_raiz(", "def limpar_metadados(", "base_bruta = pd.read_csv", "def jump_correct(", "base_recortada, comprimentos_onda = preprocessar"):
    fonte = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and marcador in "".join(c["source"]))
    exec(compile(fonte, str(ROOT / "script_final/Pipeline_Final_Classificacao.ipynb"), "exec"), ambiente)

base = ambiente["base_normalizada"]
base = base.loc[base["turno"].eq("manha")].copy()
base["dia"] = base["data_coleta"].astype(str).str.extract(r"^(D\d+)")
regioes = [(405, 435), (710, 750), (1350, 1373), (1926, 1997)]
resultados = []
for (dia, condicao), grupo in base.groupby(["dia", "condicao"]):
    for numero, (inicio, fim) in enumerate(regioes, 1):
        x = np.arange(inicio, fim + 1, dtype=float)
        y = grupo[[str(int(w)) for w in x]].mean().to_numpy(dtype=float)
        desvio = y - np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        pico = int(np.argmax(np.abs(desvio)))
        sinal = 1 if desvio[pico] >= 0 else -1
        d = sinal * desvio
        metade = d[pico] / 2
        esquerda = pico
        while esquerda > 0 and d[esquerda] > metade:
            esquerda -= 1
        direita = pico
        while direita < len(x) - 1 and d[direita] > metade:
            direita += 1
        if d[pico] <= 1e-12 or esquerda == pico or direita == pico:
            x_esq = x_dir = np.nan
        else:
            x_esq = np.interp(metade, [d[esquerda], d[esquerda + 1]], [x[esquerda], x[esquerda + 1]])
            x_dir = np.interp(metade, [d[direita], d[direita - 1]], [x[direita], x[direita - 1]])
        resultados.append(dict(dia=dia, condicao=condicao, n=len(grupo), regiao=numero, extremo_esquerdo_nm=inicio, extremo_direito_nm=fim, tipo="pico" if sinal == 1 else "vale", centro_nm=x[pico], meia_altura_esquerda_nm=x_esq, meia_altura_direita_nm=x_dir, fwhm_nm=x_dir-x_esq))

saida = ROOT / "script_final/outputs/classificacao/fwhm_regioes_por_dia.csv"
pd.DataFrame(resultados).to_csv(saida, sep=";", decimal=",", index=False, float_format="%.3f")
print(pd.DataFrame(resultados).pivot(index=["dia", "condicao"], columns="regiao", values="fwhm_nm").round(2).to_string())
print(f"Resultados: {saida}")
