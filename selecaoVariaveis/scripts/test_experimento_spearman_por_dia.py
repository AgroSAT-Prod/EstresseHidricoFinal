from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selecaoVariaveis"))
sys.path.insert(0, str(ROOT / "reducaoColinearidade"))

from experimento_spearman_por_dia import (  # noqa: E402
    classificar_pls,
    escolher_melhor_componente,
    escolher_representantes,
    executar_experimento,
    selecionar_top_baixa_colinearidade,
)
from reducao_colinearidade import JANELA_NM, LIMIAR_R, agrupar  # noqa: E402


def agrupar_legado(corr: np.ndarray, w: np.ndarray) -> np.ndarray:
    grupo = np.zeros(len(w), dtype=int)
    membros = [0]
    atual = 0
    for j in range(1, len(w)):
        dentro_janela = w[j] - w[membros[0]] < JANELA_NM
        correlacionada = dentro_janela and np.all(
            np.abs(corr[j, membros]) > LIMIAR_R
        )
        if correlacionada:
            membros.append(j)
        else:
            atual += 1
            membros = [j]
        grupo[j] = atual
    return grupo


class TestAgrupamento(unittest.TestCase):
    def test_padrao_reproduz_implementacao_legada(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.uniform(-1, 1, size=(12, 12))
        corr = (a + a.T) / 2
        np.fill_diagonal(corr, 1.0)
        w = np.arange(400, 412, dtype=float)
        np.testing.assert_array_equal(agrupar(corr, w), agrupar_legado(corr, w))

    def test_ligacao_completa_e_monotonicidade(self) -> None:
        corr = np.array([
            [1.00, 0.96, 0.96],
            [0.96, 1.00, 0.94],
            [0.96, 0.94, 1.00],
        ])
        w = np.array([400.0, 401.0, 402.0])
        grupos_08 = agrupar(corr, w, limiar_r=0.80)
        grupos_09 = agrupar(corr, w, limiar_r=0.90)
        grupos_095 = agrupar(corr, w, limiar_r=0.95)
        self.assertEqual(len(np.unique(grupos_08)), 1)
        self.assertEqual(len(np.unique(grupos_09)), 1)
        self.assertEqual(len(np.unique(grupos_095)), 2)
        self.assertLessEqual(len(np.unique(grupos_08)), len(np.unique(grupos_09)))
        self.assertLessEqual(len(np.unique(grupos_09)), len(np.unique(grupos_095)))

    def test_janela_e_estritamente_menor_que_dez_nm(self) -> None:
        corr = np.full((3, 3), 0.99)
        np.fill_diagonal(corr, 1.0)
        w = np.array([400.0, 409.0, 410.0])
        grupos = agrupar(corr, w, limiar_r=0.95, janela_nm=10.0)
        np.testing.assert_array_equal(grupos, np.array([0, 0, 1]))


class TestRepresentantes(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.array([400.0, 401.0, 402.0])
        self.grupos = np.zeros(3, dtype=int)
        self.corr = np.array([
            [1.00, 0.81, 0.81],
            [0.81, 1.00, 0.99],
            [0.81, 0.99, 1.00],
        ])

    def test_empate_por_faixa_reproduz_isclose_legado(self) -> None:
        p = np.array([1e-12, 5e-9, 1e-6])
        reps = escolher_representantes(self.corr, self.w, self.grupos, p)
        np.testing.assert_array_equal(reps, np.array([False, True, False]))

    def test_p_fora_da_faixa_nao_e_tratado_como_empate(self) -> None:
        p = np.array([1e-12, 2e-8, 1e-6])
        reps = escolher_representantes(self.corr, self.w, self.grupos, p)
        np.testing.assert_array_equal(reps, np.array([True, False, False]))

    def test_empate_exato_e_resolvido_pelo_medoide(self) -> None:
        p = np.array([1e-4, 1e-4, 1e-3])
        reps = escolher_representantes(self.corr, self.w, self.grupos, p)
        np.testing.assert_array_equal(reps, np.array([False, True, False]))

    def test_empate_de_centralidade_prefere_menor_banda(self) -> None:
        corr = np.full((3, 3), 0.9)
        np.fill_diagonal(corr, 1.0)
        reps = escolher_representantes(
            corr, self.w, self.grupos, np.full(3, np.nan)
        )
        np.testing.assert_array_equal(reps, np.array([True, False, False]))


class TestPLSDA(unittest.TestCase):
    def test_limiar_de_predicao_e_meio(self) -> None:
        predicoes = np.array([-0.1, 0.0, 0.49, 0.5, 0.9])
        np.testing.assert_array_equal(
            classificar_pls(predicoes), np.array([0, 0, 0, 1, 1])
        )

    def test_empate_escolhe_menor_numero_de_componentes(self) -> None:
        validacao = pd.DataFrame({
            "n_componentes": [1, 2, 3],
            "balanced_accuracy_cv_diagnostica": [0.7, 0.8, 0.8],
        })
        self.assertEqual(escolher_melhor_componente(validacao), 2)


class TestSelecaoTop(unittest.TestCase):
    @staticmethod
    def candidatas(w: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "banda_nm": w.astype(int),
            "significativa": True,
            "vip_pls_da": np.arange(len(w), 0, -1, dtype=float),
            "p_valor": np.linspace(0.001, 0.003, len(w)),
        })

    def test_limiar_e_correlacao_maxima_aceita(self) -> None:
        w = np.array([400.0, 410.0, 420.0])
        corr = np.array([
            [1.00, 0.85, 0.20],
            [0.85, 1.00, 0.20],
            [0.20, 0.20, 1.00],
        ])
        top_08 = selecionar_top_baixa_colinearidade(
            self.candidatas(w), corr, w, correlacao_maxima=0.80
        )
        top_09 = selecionar_top_baixa_colinearidade(
            self.candidatas(w), corr, w, correlacao_maxima=0.90
        )
        self.assertEqual(top_08["banda_nm"].tolist(), [400, 420])
        self.assertEqual(top_09["banda_nm"].tolist(), [400, 410, 420])

    def test_correlacao_igual_ao_limiar_nao_e_aceita(self) -> None:
        w = np.array([400.0, 410.0])
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        top = selecionar_top_baixa_colinearidade(
            self.candidatas(w), corr, w, correlacao_maxima=0.80
        )
        self.assertEqual(top["banda_nm"].tolist(), [400])

    def test_separacao_minima_de_dez_nm(self) -> None:
        w = np.array([400.0, 405.0, 410.0])
        corr = np.full((3, 3), 0.1)
        np.fill_diagonal(corr, 1.0)
        top = selecionar_top_baixa_colinearidade(
            self.candidatas(w), corr, w, correlacao_maxima=0.80
        )
        self.assertEqual(top["banda_nm"].tolist(), [400, 410])


class TestIntegracao(unittest.TestCase):
    def test_tres_correlacoes_maximas_geram_tabelas_longas(self) -> None:
        rng = np.random.default_rng(7)
        blocos = np.repeat(["B1", "B2", "B3", "B4"], 4)
        condicoes = np.tile(["IRRIG", "IRRIG", "NIRRIG", "NIRRIG"], 4)
        y = (condicoes == "NIRRIG").astype(float)
        espectro = rng.normal(size=(len(blocos), 6)) + y[:, None] * np.linspace(0.2, 1.2, 6)
        w = np.arange(400, 406, dtype=float)
        meta = pd.DataFrame({
            "genotipo": "BR16",
            "dia": "D02",
            "turno": "manha",
            "bloco": blocos,
            "condicao": condicoes,
        })
        estatisticas = pd.DataFrame({
            "genotipo": "BR16",
            "dia": "D02",
            "banda_nm": w.astype(int),
            "p_valor": np.linspace(0.001, 0.006, len(w)),
            "q_fdr": 0.01,
            "epsilon2": 0.2,
            "delta_cliff": np.linspace(0.2, 0.7, len(w)),
        })
        with patch(
            "experimento_spearman_por_dia.agrupar", wraps=agrupar
        ) as agrupar_monitorado:
            tabelas = executar_experimento(
                meta, espectro, w, estatisticas,
                genotipos=("BR16",),
                correlacoes_maximas=(0.80, 0.90, 0.95),
            )
        self.assertEqual(
            [c.kwargs["limiar_r"] for c in agrupar_monitorado.call_args_list],
            [0.80, 0.90, 0.95],
        )
        resumo = tabelas["resumo_cenarios.csv"]
        detalhe = tabelas["bandas_detalhadas.csv"]
        top = tabelas["top5_bandas.csv"]
        comparacao = tabelas["comparacao_limiares.csv"]

        self.assertEqual(len(resumo), 3)
        self.assertEqual(len(detalhe), 18)
        self.assertEqual(len(comparacao), 3)
        self.assertTrue(
            (top.groupby("correlacao_maxima_aceita_top").size() <= 5).all()
        )
        self.assertTrue((top["q_fdr"] < 0.05).all())
        grupos = resumo.sort_values(
            "correlacao_maxima_aceita_top"
        )["grupos_spearman"]
        self.assertTrue(grupos.is_monotonic_increasing)
        self.assertTrue(
            (top["rho_abs_max_outro_top"].dropna()
             < top.loc[top["rho_abs_max_outro_top"].notna(),
                       "correlacao_maxima_aceita_top"]).all()
        )
        self.assertTrue(
            (top["distancia_min_outro_top_nm"].dropna() >= 10.0).all()
        )


if __name__ == "__main__":
    unittest.main()
