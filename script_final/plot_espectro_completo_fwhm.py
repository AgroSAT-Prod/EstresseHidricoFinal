"""Espectro completo e detalhes das quatro regiões de FWHM por dia."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PASTA = ROOT / "script_final/outputs/classificacao"
notebook = json.loads((ROOT / "script_final/Pipeline_Final_Classificacao.ipynb").read_text(encoding="utf-8"))
ambiente = {}
for marcador in ("def localizar_raiz(", "def limpar_metadados(", "base_bruta = pd.read_csv", "def jump_correct(", "base_recortada, comprimentos_onda = preprocessar"):
    fonte = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and marcador in "".join(c["source"]))
    exec(compile(fonte, str(ROOT / "script_final/Pipeline_Final_Classificacao.ipynb"), "exec"), ambiente)

base = ambiente["base_normalizada"]
base = base.loc[base["turno"].eq("manha")].copy()
base["dia"] = base["data_coleta"].astype(str).str.extract(r"^(D\d+)")
resultados = pd.read_csv(PASTA / "fwhm_regioes_por_dia.csv", sep=";", decimal=",")
regioes = [(405, 435), (710, 750), (1350, 1373), (1926, 1997)]
condicoes = [("IRRIG", "Irrigado", "#2166ac"), ("NIRRIG", "Não irrigado", "#d95f02")]
x = ambiente["comprimentos_onda"]
colunas = [str(int(w)) for w in x]
dias = sorted(base["dia"].unique())
medias = {(dia, cond): grupo[colunas].mean().to_numpy() for (dia, cond), grupo in base.groupby(["dia", "condicao"])}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def espectro(eixo, dia):
    for cond, nome, cor in condicoes:
        eixo.plot(x, medias[dia, cond], color=cor, lw=1.5, label=nome)
    for numero, (inicio, fim) in enumerate(regioes, 1):
        eixo.axvspan(inicio, fim, color="#9a8c55", alpha=0.18)
        eixo.text((inicio + fim) / 2, 0.97, f"R{numero}", transform=eixo.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        for indice, (cond, _, cor) in enumerate(condicoes):
            r = resultados.loc[(resultados.dia == dia) & (resultados.condicao == cond) & (resultados.regiao == numero)].iloc[0]
            eixo.plot([r.meia_altura_esquerda_nm, r.meia_altura_direita_nm], [0.06 + indice * 0.045] * 2,
                      transform=eixo.get_xaxis_transform(), color=cor, lw=3, marker="|", markersize=6)
    eixo.set_xlim(400, 2450)
    eixo.set_title(dia, loc="left", fontweight="bold")
    eixo.grid(alpha=0.18)

fig, eixos = plt.subplots(4, 2, figsize=(16, 15), sharex=True, sharey=True, layout="constrained")
for eixo, dia in zip(eixos.flat, dias):
    espectro(eixo, dia)
eixos.flat[-1].axis("off")
eixos.flat[-1].legend(handles=[Line2D([], [], color=c, label=n) for _, n, c in condicoes] + [Patch(facecolor="#9a8c55", alpha=0.18, label="Regiões analisadas")], loc="upper left", frameon=False)
eixos.flat[-1].text(0.03, 0.6, "Barras na base dos painéis: intervalos de FWHM.\n\nR1: 405–435 nm     R2: 710–750 nm\nR3: 1350–1373 nm   R4: 1926–1997 nm\n\nManhã; três genótipos reunidos.\nCorreção de emendas + Savitzky–Golay + SNV.", transform=eixos.flat[-1].transAxes, va="top", linespacing=1.7)
fig.suptitle("Espectro completo com os intervalos de FWHM por dia", fontsize=18)
fig.supxlabel("Comprimento de onda (nm)")
fig.supylabel("Resposta espectral normalizada (SNV)")
fig.savefig(PASTA / "espectro_completo_fwhm.png", dpi=200)
plt.close(fig)

with PdfPages(PASTA / "espectro_completo_fwhm_detalhes.pdf") as pdf:
    for dia in dias:
        fig = plt.figure(figsize=(16, 9), layout="constrained")
        grade = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])
        eixo = fig.add_subplot(grade[0, :])
        espectro(eixo, dia)
        eixo.legend(loc="upper right")
        eixo.set_ylabel("Resposta SNV")
        for numero, (inicio, fim) in enumerate(regioes, 1):
            ax = fig.add_subplot(grade[1, numero - 1])
            for cond, nome, cor in condicoes:
                y = medias[dia, cond]
                mascara = (x >= inicio) & (x <= fim)
                xx, yy = x[mascara], y[mascara]
                fundo = np.interp(xx, [inicio, fim], [yy[0], yy[-1]])
                r = resultados.loc[(resultados.dia == dia) & (resultados.condicao == cond) & (resultados.regiao == numero)].iloc[0]
                altura = np.interp(r.centro_nm, xx, yy - fundo) / 2
                limites = np.array([r.meia_altura_esquerda_nm, r.meia_altura_direita_nm])
                ax.plot(xx, yy, color=cor, lw=2, label=f"{nome}: {r.fwhm_nm:.2f} nm")
                ax.plot(xx, fundo, color=cor, ls=":", alpha=0.6)
                ax.plot(limites, np.interp(limites, xx, fundo) + altura, color=cor, ls="--", marker="|", markersize=10)
            ax.set_title(f"R{numero} · {inicio}–{fim} nm")
            ax.set_xlabel("Comprimento de onda (nm)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{dia} — Espectro completo e FWHM nas quatro regiões", fontsize=17)
        fig.supxlabel("Pontilhado: linha de base entre extremos. Tracejado: meia altura em relação à linha de base.", fontsize=10)
        fig.savefig(PASTA / f"espectro_completo_fwhm_{dia}.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
print("Salvos: espectro_completo_fwhm.png, PDF e sete figuras por dia.")
