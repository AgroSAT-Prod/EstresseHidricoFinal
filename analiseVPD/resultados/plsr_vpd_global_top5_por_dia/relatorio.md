# PLSR — VPD global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_vpd | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 31 | 714 | 0.0554 | 0.0554 | 0.4376 | 0.2720 | nan | True |
| 2 | 3 | 112 | 1520 | 0.0547 | 0.0547 | 0.9381 | 0.4758 | 0.0741 | True |
| 3 | 12 | 98 | 1375 | -0.0485 | 0.0485 | -2.3780 | -0.3352 | 0.4590 | True |
| 4 | 28 | 187 | 2272 | -0.0364 | 0.0364 | -0.5263 | 0.1395 | 0.7892 | True |
| 5 | 35 | 150 | 1902 | -0.0338 | 0.0338 | -0.5824 | 0.3762 | 0.6661 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 5 | 0.6367 | 0.5142 | 0.5800 | 0.5529 | nan |
| 23/02 | 24 | 4 | 0.5619 | 0.1652 | -0.4525 | 0.3009 | D02M |
| 24/02 | 24 | 4 | 0.7912 | 0.3826 | 0.6119 | 0.5216 | D03M |
| 25/02 | 24 | 3 | 0.7328 | 0.2809 | 0.6462 | 0.3232 | D04M |
| 26/02 | 24 | 5 | 0.8237 | 0.4598 | 0.6367 | 0.6600 | D05M |
| 27/02 | 24 | 5 | 0.9148 | 0.1712 | 0.8420 | 0.2332 | D06M |
| 02/03 | 24 | 5 | 0.7937 | 0.4489 | 0.5275 | 0.6794 | D09M |
| 03/03 | 24 | 5 | 0.7687 | 0.3173 | 0.4638 | 0.4831 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
