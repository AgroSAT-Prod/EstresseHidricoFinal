# PLSR — CHL B global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_chl_b | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 15 | 551 | -0.0772 | 0.0772 | -1.1612 | -0.3380 | nan | True |
| 2 | 3 | 133 | 1725 | -0.0762 | 0.0762 | -3.8152 | -0.3488 | 0.2559 | True |
| 3 | 5 | 97 | 1365 | 0.0759 | 0.0759 | 6.5919 | 0.2029 | 0.3657 | True |
| 4 | 18 | 83 | 1233 | 0.0556 | 0.0556 | 2.5433 | 0.2777 | 0.5307 | True |
| 5 | 27 | 116 | 1564 | 0.0512 | 0.0512 | 2.0129 | -0.1311 | 0.7523 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 5 | 0.6338 | 0.7634 | 0.5837 | 0.8141 | nan |
| 23/02 | 24 | 4 | 0.6946 | 0.6672 | 0.4789 | 0.8716 | D02M |
| 24/02 | 24 | 2 | 0.6889 | 0.7068 | 0.6057 | 0.7958 | D03M |
| 25/02 | 24 | 5 | 0.8459 | 0.4622 | 0.6556 | 0.6910 | D04M |
| 26/02 | 24 | 5 | 0.7393 | 0.6129 | 0.4988 | 0.8497 | D05M |
| 27/02 | 24 | 3 | 0.5600 | 0.7499 | 0.3955 | 0.8789 | D06M |
| 02/03 | 24 | 5 | 0.7291 | 0.7216 | 0.3989 | 1.0749 | D09M |
| 03/03 | 24 | 5 | 0.7516 | 0.5667 | 0.2439 | 0.9888 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
