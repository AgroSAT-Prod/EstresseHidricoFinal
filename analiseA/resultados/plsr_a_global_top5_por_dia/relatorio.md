# PLSR — Taxa fotossintética (A) global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_a | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 98 | 1375 | 0.0448 | 0.0448 | 16.8021 | 0.3189 | nan | True |
| 2 | 3 | 31 | 714 | -0.0400 | 0.0400 | -2.4159 | -0.2044 | 0.4590 | True |
| 3 | 5 | 96 | 1355 | 0.0380 | 0.0380 | 17.8903 | 0.6289 | 0.6145 | True |
| 4 | 7 | 112 | 1515 | -0.0360 | 0.0360 | -4.5040 | -0.5538 | 0.5056 | True |
| 5 | 18 | 143 | 1832 | 0.0273 | 0.0273 | 5.5758 | -0.1718 | 0.7896 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 5 | 0.6746 | 3.7199 | 0.6123 | 4.0602 | nan |
| 23/02 | 24 | 4 | 0.4042 | 2.0217 | -0.3538 | 3.0475 | D02M |
| 24/02 | 24 | 2 | 0.8557 | 3.3636 | 0.7729 | 4.2204 | D03M |
| 25/02 | 24 | 1 | 0.4015 | 3.1495 | 0.2939 | 3.4208 | D04M |
| 26/02 | 24 | 3 | 0.7171 | 4.3201 | 0.6308 | 4.9353 | D05M |
| 27/02 | 24 | 5 | 0.8562 | 1.8606 | 0.6973 | 2.6995 | D06M |
| 02/03 | 24 | 4 | 0.8165 | 3.1514 | 0.7139 | 3.9355 | D09M |
| 03/03 | 24 | 5 | 0.6629 | 1.9127 | 0.3240 | 2.7084 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
