"""Curvas médias por dia usando o pré-processamento da classificação."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
notebook = json.loads((ROOT / "script_final/Pipeline_Final_Classificacao.ipynb").read_text(encoding="utf-8"))
ambiente = {}
# As células são identificadas pelo conteúdo para reutilizar a própria pipeline.
for marcador in ("def localizar_raiz(", "def limpar_metadados(", "base_bruta = pd.read_csv", "def jump_correct(", "base_recortada, comprimentos_onda = preprocessar"):
    fonte = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and marcador in "".join(c["source"]))
    exec(compile(fonte, str(ROOT / "script_final/Pipeline_Final_Classificacao.ipynb"), "exec"), ambiente)

base = ambiente["base_normalizada"]
base = base.loc[base["turno"].eq("manha")].copy()
base["dia"] = base["data_coleta"].astype(str).str.extract(r"^(D\d+)")
ondas = np.arange(1927, 1998)
colunas = [str(w) for w in ondas]
dias = sorted(base["dia"].dropna().unique())
fig, eixos = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True, layout="constrained")
for eixo, dia in zip(eixos.flat, dias):
    for condicao, nome, cor in [("IRRIG", "Irrigado", "#2166ac"), ("NIRRIG", "Não irrigado", "#d95f02")]:
        valores = base.loc[(base["dia"] == dia) & (base["condicao"] == condicao), colunas].to_numpy(dtype=float)
        if len(valores):
            eixo.plot(ondas, valores.mean(axis=0), color=cor, label=f"{nome} (n={len(valores)})")
    eixo.set_title(dia)
    eixo.grid(alpha=0.2)
    eixo.legend(fontsize=8)
for eixo in list(eixos.flat)[len(dias):]:
    eixo.set_visible(False)
fig.supxlabel("Comprimento de onda (nm)")
fig.supylabel("Resposta espectral normalizada (SNV)")
fig.suptitle("Curvas espectrais médias por dia — 1927 a 1997 nm\nLeituras da manhã; genótipos reunidos", fontsize=14)
saida = ROOT / "script_final/outputs/classificacao/curva_espectral_1927_1997.png"
saida.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(saida, dpi=200)
print(f"Gráfico salvo em: {saida}")
