 1. Dados e pré-processamento
      - Apenas leituras do turno da manhã, presente nos sete dias avaliados (D02, D03, D04, D05, D06, D09 e D10).
      - Espectros entre 400 e 2450 nm: 2.051 bandas.
      - Correção dos saltos entre detectores (~1000 e ~1800 nm), tratamento/interpolação de valores inválidos, suavização Savitzky–Golay (janela 11, polinômio de ordem 2) e
        normalização SNV por espectro.

  2. Comparações feitas, dia a dia
     Dentro de cada combinação dia × condição (IRRIG ou NIRRIG), foram comparadas as três duplas de genótipos:
      - BR16 × CD202
      - BR16 × EMB48
      - CD202 × EMB48

     Portanto, a lista não representa “irrigado versus não irrigado”. Ela mostra as bandas que mais separam cada genótipo dos demais, mantendo condição e dia fixos.

  3. Teste por banda
     Para cada par de genótipos e cada banda foi aplicado Mann–Whitney bilateral.
      - Usou-se o teste exato quando não havia empates;
      - Quando havia empates, usou-se a aproximação assintótica;
      - O Kruskal–Wallis para dois grupos também foi calculado e salvo como referência, mas o teste ativo para selecionar as bandas foi Mann–Whitney;
      - Cada grupo comparado tinha 32 leituras (4 blocos × 8 varreduras).

  4. Correção para múltiplas bandas
     Os p-valores foram ajustados por Benjamini–Hochberg (FDR) dentro de cada contraste dia × condição × par de genótipos, considerando as 2.051 bandas como uma família de
     testes. O arquivo final contém tanto p_valor quanto q_fdr.

  5. Regra do top 5
     Para cada dia × genótipo × condição:
      - Para cada banda, foi mantido o contraste contra o outro genótipo com menor p_valor;
      - Foram filtradas bandas com p bruto ≤ 0,05;
      - As cinco menores probabilidades foram selecionadas e ordenadas.

  Importante: o corte que definiu o top 5 foi o p bruto, não o q_fdr. Porém, nos resultados apresentados, todas as bandas também permaneceram significativas após FDR: o maior
  q_fdr entre elas foi 0,00308.

  A ressalva estatística é que as 32 varreduras foram tratadas como observações no Mann–Whitney, embora provenham de quatro blocos experimentais. Logo, os resultados são
  excelentes para localizar regiões espectrais discriminantes, mas a inferência estritamente experimental deve considerar a dependência das oito varreduras dentro de cada
  bloco.  


   Dia    Genótipo    Irrigado                        Não irrigado
  ━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   D02    BR16        708, 709, 710, 711, 712         1917, 1918, 1919, 1920, 1915
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       1902, 1901, 1903, 1900, 1899    1908, 1909, 1907, 1906, 1905
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       708, 709, 710, 711, 712         1917, 1918, 1919, 1920, 1915
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
   D03    BR16        532, 533, 534, 535, 536         702, 703, 704, 705, 706
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       532, 533, 534, 535, 536         702, 703, 704, 705, 706
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       715, 716, 714, 717, 718         718, 719, 720, 721, 722
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
   D04    BR16        523, 524, 525, 526, 527         703, 704, 705, 706, 707
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       523, 524, 525, 526, 527         703, 704, 705, 706, 707
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       710, 711, 712, 713, 714         1902, 1903, 1905, 1904, 1901
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
   D05    BR16        1731, 1732, 1733, 1734, 1735    1939, 1936, 1937, 1938, 1940
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       1382, 1383, 1381, 1380, 1379    1939, 1936, 1937, 1938, 1940
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       1731, 1732, 1733, 1734, 1735    1750, 1751, 1752, 1753, 1749
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
   D06    BR16        719, 720, 721, 722, 723         717, 718, 719, 720, 721
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       1398, 1399, 1400, 1401, 1402    723, 722, 724, 725, 726
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       719, 720, 721, 722, 723         717, 718, 719, 720, 721
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
   D09    BR16        2053, 2051, 2052, 2054, 2056    711, 712, 713, 714, 715
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       2053, 2051, 2052, 2054, 2056    715, 716, 717, 718, 719
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       686, 687, 683, 684, 685         711, 712, 713, 714, 715
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
   D10    BR16        698, 699, 697, 2032, 700        713, 714, 715, 716, 717
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          CD202       698, 1380, 1381, 1383, 1379     1333, 1334, 1335, 1336, 1337
  ─────  ──────────  ──────────────────────────────  ──────────────────────────────
          EMB48       1380, 1381, 1383, 1379, 1382    713, 714, 715, 716, 717
