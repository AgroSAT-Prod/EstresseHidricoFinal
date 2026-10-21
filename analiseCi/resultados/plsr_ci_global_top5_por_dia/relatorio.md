# PLSR — Ci global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_ci | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 154 | 1942 | -0.0337 | 0.0337 | -35.5782 | -0.6486 | nan | True |
| 2 | 4 | 98 | 1375 | 0.0317 | 0.0317 | 105.8422 | 0.1996 | 0.0673 | True |
| 3 | 9 | 144 | 1842 | 0.0286 | 0.0286 | 46.8452 | -0.1442 | 0.4389 | True |
| 4 | 12 | 31 | 714 | -0.0280 | 0.0280 | -15.0185 | -0.1783 | 0.4590 | True |
| 5 | 17 | 96 | 1355 | 0.0259 | 0.0259 | 108.4951 | 0.5097 | 0.6145 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 5 | 0.5039 | 40.8288 | 0.4500 | 42.9910 | nan |
| 23/02 | 24 | 3 | 0.5283 | 30.2718 | 0.2153 | 39.0439 | D02M |
| 24/02 | 24 | 1 | 0.4469 | 57.1110 | 0.2797 | 65.1720 | D03M |
| 25/02 | 24 | 4 | 0.6553 | 27.0872 | 0.4334 | 34.7297 | D04M |
| 26/02 | 24 | 1 | 0.5304 | 44.0090 | 0.3120 | 53.2692 | D05M |
| 27/02 | 24 | 5 | 0.8649 | 22.1265 | 0.7716 | 28.7744 | D06M |
| 02/03 | 24 | 5 | 0.7661 | 27.8230 | 0.4779 | 41.5743 | D09M |
| 03/03 | 24 | 1 | 0.5561 | 23.5553 | 0.4800 | 25.4954 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
