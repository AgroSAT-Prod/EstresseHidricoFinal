# PLSR — CHL Total global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_chl_total | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 133 | 1725 | -0.0815 | 0.0815 | -8.5341 | -0.3365 | nan | True |
| 2 | 4 | 96 | 1355 | 0.0756 | 0.0756 | 14.4043 | 0.3156 | 0.0835 | True |
| 3 | 15 | 83 | 1233 | 0.0625 | 0.0625 | 5.9804 | 0.3157 | 0.6951 | True |
| 4 | 21 | 34 | 735 | -0.0597 | 0.0597 | -2.8254 | -0.3761 | 0.3208 | True |
| 5 | 30 | 149 | 1892 | -0.0560 | 0.0560 | -2.5427 | 0.0647 | 0.7698 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 4 | 0.4663 | 1.9272 | 0.4276 | 1.9957 | nan |
| 23/02 | 24 | 4 | 0.5359 | 1.5540 | 0.3739 | 1.8049 | D02M |
| 24/02 | 24 | 2 | 0.6137 | 1.4906 | 0.4982 | 1.6990 | D03M |
| 25/02 | 24 | 4 | 0.7992 | 1.0427 | 0.6173 | 1.4395 | D04M |
| 26/02 | 24 | 3 | 0.6443 | 1.4229 | 0.4301 | 1.8011 | D05M |
| 27/02 | 24 | 2 | 0.4395 | 1.5958 | 0.1761 | 1.9348 | D06M |
| 02/03 | 24 | 2 | 0.3958 | 2.0302 | 0.0586 | 2.5342 | D09M |
| 03/03 | 24 | 1 | 0.4584 | 1.9322 | 0.2167 | 2.3238 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
