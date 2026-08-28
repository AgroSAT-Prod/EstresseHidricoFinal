#!/usr/bin/env python3
"""Um único correlograma para a união das Top 5 VIP de todos os dias/genótipos."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT / "testeDeNormalidade"))
from shapiro_normalidade import carregar  # noqa: E402

ENTRADA = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "plsda_vip_5seeds_kfold4"
SAIDA = ROOT / "analisePorDia" / "resultados" / "dataset_gerado" / "correlograma_top5_vip_consolidado"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    top5 = pd.read_csv(ENTRADA / "top5_bandas_vip_por_dia_genotipo.csv", sep=";")
    bandas = np.sort(top5.banda_nm.astype(int).unique())
    meta, espectro, w = carregar("normalizado", turno="manha")
    posicao = {int(nm): i for i, nm in enumerate(w.astype(int))}
    X = espectro[:, [posicao[int(b)] for b in bandas]]
    corr = pd.DataFrame(X, columns=bandas).corr(method="spearman")
    corr.to_csv(SAIDA / "correlacao_spearman_top5_consolidada.csv", sep=";", index_label="banda_nm")

    origem = (top5.groupby("banda_nm", as_index=False)
              .agg(ocorrencias=("banda_nm", "size"),
                   genotipos=("genotipo", lambda x: ", ".join(sorted(set(x)))),
                   dias=("dia", lambda x: ", ".join(sorted(set(x)))))
              .sort_values("banda_nm"))
    origem.to_csv(SAIDA / "origem_das_bandas_consolidadas.csv", sep=";", index=False)

    fig, ax = plt.subplots(figsize=(25, 22))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(range(len(bandas)), bandas, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(bandas)), bandas, fontsize=5.5)
    ax.set_xlabel("Comprimento de onda (nm)"); ax.set_ylabel("Comprimento de onda (nm)")
    ax.set_title("Correlograma consolidado: união das Top 5 VIP de todos os genótipos e dias\nSpearman; todas as leituras da manhã", fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=.026, pad=.02); cbar.set_label("Correlação de Spearman (ρ)")
    fig.tight_layout()
    fig.savefig(SAIDA / "correlograma_top5_vip_consolidado.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAIDA / "correlograma_top5_vip_consolidado.pdf", bbox_inches="tight")
    plt.close(fig)

    texto = "\n".join([
        "# Correlograma consolidado das Top 5 VIP",
        "",
        f"Foram reunidas as 105 seleções Top 5 VIP (3 genótipos × 7 dias), que correspondem a **{len(bandas)} bandas físicas únicas** após remover repetições. A matriz é uma única correlação de Spearman calculada sobre todas as {len(meta)} leituras da manhã, independentemente de genótipo, dia e condição.",
        "",
        "`origem_das_bandas_consolidadas.csv` preserva os genótipos e dias que selecionaram cada banda; `correlacao_spearman_top5_consolidada.csv` contém a matriz numérica completa.",
    ]) + "\n"
    (SAIDA / "relatorio_correlograma_consolidado.md").write_text(texto, encoding="utf-8")
    pd.DataFrame([{"metodo": "Spearman", "turno": "manha", "leituras": len(meta), "selecoes_top5": len(top5), "bandas_unicas": len(bandas), "fonte": str(ENTRADA / "top5_bandas_vip_por_dia_genotipo.csv")}]).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)
    print(f"{len(top5)} seleções reunidas em {len(bandas)} bandas únicas; {len(meta)} leituras")


if __name__ == "__main__":
    main()
