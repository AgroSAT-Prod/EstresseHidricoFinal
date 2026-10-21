# PLSR — Condutância estomática (gs) global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_gs | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32 | 715 | -0.0502 | 0.0502 | -0.1216 | -0.2068 | nan | True |
| 2 | 3 | 143 | 1832 | 0.0465 | 0.0465 | 0.3771 | -0.1849 | 0.2031 | True |
| 3 | 4 | 114 | 1535 | -0.0447 | 0.0447 | -0.2709 | -0.5472 | 0.7912 | True |
| 4 | 23 | 133 | 1725 | 0.0325 | 0.0325 | 0.3347 | 0.0234 | 0.5647 | True |
| 5 | 37 | 15 | 559 | -0.0274 | 0.0274 | -0.0865 | 0.2670 | 0.7119 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 4 | 0.6309 | 0.1576 | 0.5835 | 0.1675 | nan |
| 23/02 | 24 | 4 | 0.3956 | 0.1451 | -0.2188 | 0.2061 | D02M |
| 24/02 | 24 | 3 | 0.6834 | 0.1572 | 0.4497 | 0.2072 | D03M |
| 25/02 | 24 | 5 | 0.7017 | 0.0894 | 0.4797 | 0.1181 | D04M |
| 26/02 | 24 | 4 | 0.8248 | 0.1081 | 0.7518 | 0.1287 | D05M |
| 27/02 | 24 | 5 | 0.8362 | 0.1075 | 0.6612 | 0.1547 | D06M |
| 02/03 | 24 | 4 | 0.8342 | 0.1251 | 0.6352 | 0.1856 | D09M |
| 03/03 | 24 | 2 | 0.3766 | 0.2089 | -0.1688 | 0.2861 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
