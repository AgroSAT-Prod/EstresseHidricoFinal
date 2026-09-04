#!/usr/bin/env python3
"""Curvas espectrais de todas as leituras, coloridas por genotipo.

Onde `plot_preprocessamento_genotipo.py` reduz cada grupo a uma media, este
script mostra o dado cru: cada leitura do turno da manha vira uma linha, e a
cor identifica o genotipo. O que a figura responde e se o genotipo separa o
espectro por si so -- se as tres nuvens de curvas se sobrepoem, a variacao
entre leituras do mesmo material e maior que a variacao entre materiais, e
qualquer modelo vai precisar de mais que a forma bruta da curva.

Dois paineis, os mesmos estagios das outras figuras do modulo:

- Recortado (apos recorte + jump correction)
- Normalizado (apos Savitzky-Golay + SNV)

As linhas finas sao as leituras individuais; a linha grossa de cada cor e a
media do genotipo, para dar o eixo de leitura no meio da nuvem.

Uso:
    python plot_curvas_genotipo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib nao esta instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402

PASTA_SAIDA = ROOT.parent / "dispersoes"
SAIDA = PASTA_SAIDA / "curvas_espectrais_genotipo.png"

TURNO = "manha"

# Tres primeiros slots da paleta categorica de referencia -- o unico trio que
# passa a validacao com todos os pares em jogo (pior par CVD dE 9.2), que e o
# caso aqui: as tres cores aparecem sobrepostas no mesmo painel.
CORES_GENOTIPO = ["#2a78d6", "#eb6834", "#1baf7a"]

ESTAGIOS = [
    ("recortado", "Recortado (recorte + jump correction)", "Reflectancia"),
    ("normalizado", "Normalizado (Savitzky-Golay + SNV)", "Reflectancia (SNV)"),
]


def plot_painel(
    ax: plt.Axes,
    w: np.ndarray,
    espectro: np.ndarray,
    meta: pd.DataFrame,
    genotipos: list[str],
    cores: dict[str, str],
    titulo: str,
    unidade: str,
) -> None:
    """Um estagio: todas as leituras como linhas finas, media por cima."""
    # Ordem de desenho embaralhada, com semente fixa: plotar genotipo a
    # genotipo faz o ultimo cobrir os anteriores e sugere uma separacao que e
    # so ordem de desenho. Intercalando, a sobreposicao fica honesta.
    codigos = meta["genotipo"].to_numpy()
    ordem = np.random.default_rng(0).permutation(len(codigos))
    for i in ordem:
        ax.plot(w, espectro[i], color=cores[codigos[i]], linewidth=0.4, alpha=0.10)

    for genotipo in genotipos:
        mask = (meta["genotipo"] == genotipo).to_numpy()
        ax.plot(
            w,
            espectro[mask].mean(axis=0),
            color=cores[genotipo],
            linewidth=2.0,
            label=f"{genotipo} (n={int(mask.sum())})",
            solid_capstyle="round",
            # Contorno claro: nos trechos onde as tres medias se encostam, sem
            # ele as linhas viram uma faixa unica de cor indefinida.
            path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()],
        )

    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=9)
    ax.set_ylabel(unidade, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def gerar_figura(
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
) -> None:
    """Gera a figura com um painel por estagio."""
    genotipos = sorted(meta["genotipo"].unique())
    if len(genotipos) > len(CORES_GENOTIPO):
        raise SystemExit(
            f"{len(genotipos)} genotipos, mas a paleta validada tem "
            f"{len(CORES_GENOTIPO)} cores. Sobrepor mais que isso num painel so "
            "nao passa nos limites de distincao -- use small multiples."
        )
    cores = dict(zip(genotipos, CORES_GENOTIPO))

    print(f"  Genotipos: {', '.join(genotipos)}")
    for genotipo in genotipos:
        n = int((meta["genotipo"] == genotipo).sum())
        print(f"    {genotipo}: {n} leituras")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(ESTAGIOS), figsize=(7.5 * len(ESTAGIOS), 5.5))

    for ax, (chave, rotulo, unidade) in zip(axes, ESTAGIOS):
        plot_painel(ax, w, estagios[chave], meta, genotipos, cores, rotulo, unidade)

    # Legenda unica, fora dos eixos: dentro do painel ela cai justamente sobre
    # a regiao do NIR, onde estao as curvas.
    handles, rotulos = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, rotulos, loc="lower center", ncol=len(genotipos),
        frameon=False, fontsize=10,
        title="Genotipo  -  linha fina = leitura individual, linha grossa = media",
        title_fontsize=10,
    )

    fig.suptitle(
        "Curvas espectrais individuais por genotipo\n"
        f"Todas as leituras do turno da manha (n={len(meta)}), antes e depois "
        "da normalizacao",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.93])
    plt.savefig(SAIDA, dpi=150, bbox_inches="tight")
    print(f"\nFigura salva em: {SAIDA}")


def main() -> None:
    print("Gerando curvas espectrais por genotipo...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")
    gerar_figura(meta, estagios, w)
    print("Concluido.")


if __name__ == "__main__":
    main()
