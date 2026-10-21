# PLSR — Evapotranspiração (E) global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_e | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 400 | 0.0459 | 0.0459 | 0.8520 | 0.7974 | nan | True |
| 2 | 2 | 143 | 1832 | 0.0453 | 0.0453 | 3.8391 | -0.3029 | 0.6172 | True |
| 3 | 4 | 92 | 1322 | 0.0440 | 0.0440 | 5.1644 | 0.5754 | 0.7821 | True |
| 4 | 7 | 153 | 1932 | -0.0388 | 0.0388 | -2.0125 | -0.7030 | 0.7712 | True |
| 5 | 33 | 16 | 569 | -0.0255 | 0.0255 | -0.8885 | 0.4403 | 0.7773 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 3 | 0.6364 | 1.6351 | 0.5831 | 1.7508 | nan |
| 23/02 | 24 | 2 | 0.3703 | 1.0502 | -0.1605 | 1.4257 | D02M |
| 24/02 | 24 | 5 | 0.8896 | 0.9091 | 0.7928 | 1.2456 | D03M |
| 25/02 | 24 | 2 | 0.5909 | 1.2150 | 0.2835 | 1.6080 | D04M |
| 26/02 | 24 | 2 | 0.7347 | 1.7992 | 0.6897 | 1.9460 | D05M |
| 27/02 | 24 | 2 | 0.7927 | 0.9637 | 0.6495 | 1.2530 | D06M |
| 02/03 | 24 | 2 | 0.8066 | 1.3671 | 0.6941 | 1.7193 | D09M |
| 03/03 | 24 | 2 | 0.6987 | 1.1836 | 0.5589 | 1.4322 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
