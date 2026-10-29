"""Reexecuta para E as mesmas células metodológicas do notebook de predição."""

import json
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "analiseE"
NOTEBOOK = RAIZ / "script_final" / "Pipeline_Final_Predicao.ipynb"
CELULAS = (
    "importacoes",
    "limpeza-funcoes",
    "carregar-limpar",
    "preprocessamento-funcoes",
    "executar-preprocessamento",
    "fisiologia-pareamento",
    "fisiologia-funcoes",
    "fisiologia-global",
)


def executar() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    fontes = {celula.get("id"): "".join(celula.get("source", []))
              for celula in notebook["cells"]}
    ambiente = {"__name__": "__analise_e__"}
    for identificador in CELULAS:
        fonte = fontes[identificador]
        if identificador == "importacoes":
            fonte = fonte.replace("from IPython.display import display", "def display(obj): print(obj)")
        if identificador == "fisiologia-pareamento":
            original = 'ROOT / "script_final/outputs/fisiologia_vip_por_dia"'
            assert fonte.count(original) == 1, "Caminho de saída do notebook mudou"
            fonte = fonte.replace(original, "ROOT / \"analiseE\"")
        exec(compile(fonte, f"{NOTEBOOK.name}:{identificador}", "exec"), ambiente)

    ambiente["executar_plsr_vip_por_dia"]("E")
    resumo = pd.DataFrame(ambiente["resumos_fisiologia"])
    predicoes = pd.concat(ambiente["predicoes_fisiologia"], ignore_index=True)
    bandas = pd.concat(ambiente["bandas_vip_fisiologia"], ignore_index=True)
    resumo.to_csv(SAIDA / "metricas_e_por_dia.csv", sep=";", index=False)
    predicoes.to_csv(SAIDA / "predicoes_e_cv_bloco.csv", sep=";", index=False)
    bandas.to_csv(SAIDA / "bandas_e_top5_vip.csv", sep=";", index=False)
    print("\nMétricas E por data:")
    print(resumo[["data", "n", "bandas_nm", "componentes", "r2_ajuste",
                  "rmse_ajuste", "r2_cv_bloco", "rmse_cv_bloco"]].to_string(index=False))


if __name__ == "__main__":
    executar()
