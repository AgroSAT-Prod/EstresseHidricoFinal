"""Verificações do pareamento e da equivalência estatística ao CRA.

Executar: python3 -m unittest discover -s parametros_vs_indices/tests -v
"""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analisar_parametros_vs_indices as experimento


def fisiologia_exemplo():
    linhas = []
    for g, genotipo in enumerate(experimento.GENOTIPOS):
        for c, condicao in enumerate(("IRR", "NIR")):
            for bloco in range(4):
                linha = {"Genótipo": genotipo, "Condição": condicao, "Data": "D02M"}
                for p, parametro in enumerate(experimento.PARAMETROS):
                    for d, data in enumerate(experimento.COLETAS):
                        linha[f"{parametro} ({data})"] = p * 100 + d * 10 + g * 4 + c * 2 + bloco
                linhas.append(linha)
    return pd.DataFrame(linhas)


def espectros_exemplo():
    linhas = []
    for d, coleta in enumerate(experimento.COLETAS.values()):
        for g, genotipo in enumerate(experimento.GENOTIPOS):
            for c, condicao in enumerate(experimento.CONDICOES):
                for bloco in range(4):
                    for leitura in range(2):
                        linha = {"data_coleta": coleta, "turno": "manha", "genotipo": genotipo,
                                 "condicao": condicao, "bloco": f"B{bloco + 1}"}
                        for b in experimento.BANDAS:
                            linha[b] = int(b) / 5000 + (d + g + c + bloco) * .002 + leitura * .01
                        linhas.append(linha)
    return pd.DataFrame(linhas)


class PreparacaoTest(unittest.TestCase):
    def test_cabecalhos_datas_e_blocos_antes_das_ausencias(self):
        fisio = fisiologia_exemplo()
        fisio = fisio.rename(columns={"CHL TOTAL (25/02)": " CHL   TOTAL (25.02) "})
        fisio.loc[1, "CHL A (23/02)"] = np.nan
        longo = experimento.preparar_fisiologia(fisio)
        self.assertEqual(len(longo), 1512)
        bloco_ausente = longo.loc[longo["linha_excel"].eq(3)
                                  & longo["parametro"].eq("CHL A") & longo["data"].eq("23/02")].iloc[0]
        self.assertEqual(bloco_ausente["bloco"], "B2")
        self.assertTrue(pd.isna(bloco_ausente["valor"]))
        posterior = longo.loc[longo["linha_excel"].eq(4)].iloc[0]
        self.assertEqual(posterior["bloco"], "B3")
        self.assertEqual(set(longo.loc[longo["data"].eq("25/02"), "data_coleta"]), {"D04M"})
        self.assertEqual(experimento.normalizar_cabecalho(" ETR  (23.02) "), "ETR (23/02)")

    def test_coluna_obrigatoria_e_cabecalho_duplicado(self):
        fisio = fisiologia_exemplo()
        with self.assertRaisesRegex(ValueError, "CHL A"):
            experimento.preparar_fisiologia(fisio.drop(columns="CHL A (23/02)"))
        fisio[" CHL A (23.02) "] = fisio["CHL A (23/02)"]
        with self.assertRaisesRegex(ValueError, "cabeçalhos duplicados"):
            experimento.preparar_fisiologia(fisio)

    def test_estrutura_incompativel_com_quatro_repeticoes(self):
        fisio = fisiologia_exemplo()
        with self.assertRaisesRegex(ValueError, "quatro repetições"):
            experimento.preparar_fisiologia(pd.concat([fisio, fisio.iloc[[0]]], ignore_index=True))

    def test_formulas_com_valores_conhecidos(self):
        entrada = pd.DataFrame({"531": [3.], "570": [1.], "800": [8.], "670": [2.],
                                "790": [6.], "720": [2.], "970": [2.], "900": [4.],
                                "1600": [9.], "820": [3.]})
        indices = experimento.calcular_indices(entrada).iloc[0]
        esperado = {"PRI": .5, "NDVI": .6, "NDRE": .5, "WDI": .5, "MSI": 3.}
        for nome, valor in esperado.items():
            self.assertAlmostEqual(indices[nome], valor)
        entrada.loc[0, "900"] = 0
        self.assertTrue(pd.isna(experimento.calcular_indices(entrada).iloc[0]["WDI"]))

    def test_media_antes_do_indice_e_exclusao_da_tarde(self):
        espectros = espectros_exemplo().iloc[:2].copy()
        espectros["531"] = [1., 9.]
        espectros["570"] = [1., 1.]
        tarde = espectros.iloc[[0]].assign(turno="tarde", **{"531": 100.})
        media = experimento.preparar_espectros(pd.concat([espectros, tarde], ignore_index=True)).iloc[0]
        self.assertEqual(media["n_leituras"], 2)
        self.assertAlmostEqual(media["531"], 5.)
        self.assertAlmostEqual(media["PRI"], 2 / 3)
        self.assertNotAlmostEqual(media["PRI"], .4)  # Média dos índices individuais.

    def test_pareamento_independente_da_ordem_e_auditoria(self):
        fisio = experimento.preparar_fisiologia(fisiologia_exemplo())
        medias = experimento.preparar_espectros(espectros_exemplo())
        esperado, _ = experimento.parear_dados(fisio, medias)
        observado, auditoria = experimento.parear_dados(fisio, medias.sample(frac=1, random_state=42))
        pd.testing.assert_frame_equal(esperado, observado)
        self.assertEqual(len(auditoria), 168)
        self.assertTrue(auditoria["pareamento"].eq("pareado").all())
        self.assertEqual(len(observado), 1512)
        sem_um, audit = experimento.parear_dados(fisio, medias.iloc[1:])
        self.assertEqual(audit["pareamento"].eq("sem_espectro").sum(), 1)
        self.assertEqual(sem_um["PRI"].isna().sum(), 9)

    def test_chaves_duplicadas_no_pareamento(self):
        fisio = experimento.preparar_fisiologia(fisiologia_exemplo())
        medias = experimento.preparar_espectros(espectros_exemplo())
        with self.assertRaisesRegex(ValueError, "chaves duplicadas"):
            experimento.parear_dados(fisio, pd.concat([medias, medias.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "chaves duplicadas"):
            experimento.parear_dados(pd.concat([fisio, fisio.iloc[[0]]]), medias)

    def test_notacao_decimal_e_infinitos(self):
        valores = experimento.numerico(pd.Series(["1,5", "inválido", "inf", "-inf", None]))
        self.assertEqual(valores.iloc[0], 1.5)
        self.assertTrue(valores.iloc[1:].isna().all())


class EstatisticasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fisio = experimento.preparar_fisiologia(fisiologia_exemplo())
        medias = experimento.preparar_espectros(espectros_exemplo())
        cls.dados, _ = experimento.parear_dados(fisio, medias)

    def test_agrupamento_e_p_valores_iguais_ao_experimento_cra(self):
        spec = importlib.util.spec_from_file_location(
            "base_cra", experimento.ROOT / "relacaoCRA" / "analisar_indices_cra.py")
        base_cra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_cra)
        calculado = experimento.correlacoes(self.dados)
        self.assertEqual(len(calculado), 360)
        self.assertTrue(calculado.loc[calculado.tipo_analise.eq("por_data"), "n"].eq(24).all())
        self.assertTrue(calculado.loc[calculado.tipo_analise.eq("descritivo"), "n"].eq(168).all())
        self.assertEqual([c for c in calculado.columns if c.endswith("_p")], ["pearson_p", "spearman_p"])
        self.assertFalse(any(c.startswith("q_") or "fdr" in c.lower() for c in calculado.columns))
        # Executa a função original com as mesmas amostras e índices. O código
        # base é apenas importado; o main e os arquivos antigos não são alterados.
        for parametro in experimento.PARAMETROS:
            dados_cra = (self.dados.loc[self.dados.parametro.eq(parametro)]
                         .rename(columns={"valor": "CRA", "data": "data_cra"}))
            with patch.object(base_cra, "INDICES", {
                nome: {**config, "descricao": nome} for nome, config in experimento.INDICES.items()
            }):
                referencia = base_cra.correlacoes(dados_cra)
            novo = calculado.loc[calculado.parametro.eq(parametro)].reset_index(drop=True)
            for campo in ("n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "r2_linear", "inclinacao"):
                np.testing.assert_allclose(novo[campo], referencia[campo], rtol=1e-12, atol=1e-14)

    def test_exclusao_apenas_do_par_afetado(self):
        grupo = self.dados.loc[self.dados.parametro.eq("CHL A") & self.dados.data.eq("23/02")].copy()
        grupo.loc[grupo.index[0], "PRI"] = np.inf
        self.assertEqual(experimento.estatisticas_grupo(grupo, "PRI", "teste")["n"], 23)
        self.assertEqual(experimento.estatisticas_grupo(grupo, "NDVI", "teste")["n"], 24)
        grupo.loc[grupo.index[1], "valor"] = np.nan
        self.assertEqual(experimento.estatisticas_grupo(grupo, "PRI", "teste")["n"], 22)
        self.assertEqual(experimento.estatisticas_grupo(grupo, "NDVI", "teste")["n"], 23)

    def test_minimo_oito_antes_das_chamadas_estatisticas(self):
        grupo = self.dados.iloc[:7]
        with patch.object(experimento, "pearsonr") as pearson, patch.object(experimento, "spearmanr") as spearman:
            with self.assertRaisesRegex(ValueError, "CHL A.*mínimo 8"):
                experimento.estatisticas_grupo(grupo, "PRI", "CHL A × PRI, 23/02")
            pearson.assert_not_called()
            spearman.assert_not_called()

    def test_variaveis_constantes(self):
        grupo = self.dados.loc[self.dados.parametro.eq("CHL A") & self.dados.data.eq("23/02")].copy()
        for campo, status in (("valor", "parametro_constante"), ("PRI", "indice_constante")):
            constante = grupo.assign(**{campo: 1.})
            resultado = experimento.estatisticas_grupo(constante, "PRI", "teste")
            self.assertEqual(resultado["status"], status)
            self.assertTrue(all(pd.isna(resultado[m]) for m in experimento.METRICAS))

class GraficosTest(unittest.TestCase):
    def test_uma_imagem_por_indice_com_datas_pontos_circulares_e_sem_retas(self):
        fisio = experimento.preparar_fisiologia(fisiologia_exemplo())
        medias = experimento.preparar_espectros(espectros_exemplo())
        dados, _ = experimento.parear_dados(fisio, medias)
        dados = dados.loc[dados.parametro.eq("CHL A")]
        caminhos = []

        def conferir(fig, caminho):
            try:
                caminhos.append(caminho)
                self.assertEqual(len(fig.axes), 8)
                indice = caminho.name.upper()
                marcador = MarkerStyle("o")
                circulo = marcador.get_path().transformed(marcador.get_transform())
                for ax, (data, coleta) in zip(fig.axes, experimento.COLETAS.items()):
                    self.assertEqual(ax.get_title(), f"{data} · {coleta}")
                    self.assertEqual(len(ax.collections), 2)
                    self.assertEqual(len(ax.lines), 0)
                    self.assertFalse(ax.texts)
                    self.assertTrue(ax.get_xticklabels())
                    self.assertTrue(ax.get_yticklabels())
                    self.assertTrue(ax.get_shared_x_axes().joined(ax, fig.axes[0]))
                    self.assertTrue(ax.get_shared_y_axes().joined(ax, fig.axes[0]))
                    for pontos, condicao in zip(ax.collections, experimento.CONDICOES):
                        esperado = dados.loc[dados.data.eq(data) & dados.condicao.eq(condicao)]
                        self.assertEqual(len(esperado), 12)
                        self.assertEqual(set(esperado.genotipo), set(experimento.GENOTIPOS))
                        np.testing.assert_allclose(pontos.get_offsets(),
                                                   esperado[[indice, "valor"]].to_numpy(float))
                        np.testing.assert_allclose(pontos.get_paths()[0].vertices, circulo.vertices)
                self.assertFalse(fig.axes[7].texts)
            finally:
                experimento.plt.close(fig)

        with patch.object(experimento, "salvar_figura", side_effect=conferir):
            experimento.graficos_agrupados(dados, "CHL A", Path("saida"))
        self.assertEqual(len(caminhos), 5)
        self.assertEqual(len(set(caminhos)), 5)
        self.assertIn(Path("saida/agrupados/chl_a/pri"), caminhos)
        self.assertIn(Path("saida/agrupados/chl_a/msi"), caminhos)

    def test_limpeza_remove_so_os_graficos_antigos_conhecidos(self):
        with TemporaryDirectory() as pasta:
            saida = Path(pasta)
            antigo_agrupado = saida / "agrupados/chl_a.png"
            antigo_separado = saida / "separados/BR16/chl_a_pri.pdf"
            antigo_paineis = saida / "agrupados/chl_a/d02m_pri.png"
            antigo_data = saida / "agrupados/chl_a/d02m.png"
            novo = saida / "agrupados/chl_a/pri.png"
            pessoal = saida / "separados/BR16/anotacoes.txt"
            for arquivo in (antigo_agrupado, antigo_separado, antigo_paineis, antigo_data, novo, pessoal):
                arquivo.parent.mkdir(parents=True, exist_ok=True)
                arquivo.write_text("teste")
            experimento.remover_graficos_antigos(saida)
            self.assertFalse(antigo_agrupado.exists())
            self.assertFalse(antigo_separado.exists())
            self.assertFalse(antigo_paineis.exists())
            self.assertFalse(antigo_data.exists())
            self.assertTrue(novo.exists())
            self.assertTrue(pessoal.exists())


class DadosReaisTest(unittest.TestCase):
    def test_cobertura_com_os_arquivos_do_repositorio(self):
        dados, medias, auditoria = experimento.preparar_dados()
        self.assertEqual(len(medias), 168)
        self.assertEqual(len(dados), 1512)
        self.assertTrue(dados["valor"].notna().all())
        self.assertTrue(dados[list(experimento.INDICES)].notna().all().all())
        self.assertTrue(auditoria["pareamento"].eq("pareado").all())
        self.assertEqual(medias["n_leituras"].value_counts().to_dict(), {8: 167, 12: 1})
        tabela = experimento.correlacoes(dados)
        self.assertEqual(len(tabela), 360)
        self.assertTrue(tabela["status"].eq("ok").all())
        self.assertTrue(tabela.loc[tabela.tipo_analise.eq("por_data"), "n"].eq(24).all())


if __name__ == "__main__":
    unittest.main()
