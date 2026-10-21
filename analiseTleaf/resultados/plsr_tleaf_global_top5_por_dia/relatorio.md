# PLSR — Temperatura foliar (Tleaf) global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_tleaf | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 152 | 1913 | -0.1163 | 0.1163 | -6.8239 | -0.1852 | nan | True |
| 2 | 3 | 97 | 1374 | -0.1037 | 0.1037 | -17.4865 | -0.2588 | 0.0521 | True |
| 3 | 4 | 35 | 749 | -0.1021 | 0.1021 | -5.9923 | -0.0122 | 0.6879 | True |
| 4 | 11 | 93 | 1325 | 0.0839 | 0.0839 | 10.7533 | 0.2820 | 0.6442 | True |
| 5 | 22 | 99 | 1388 | -0.0702 | 0.0702 | -6.2074 | -0.3155 | 0.7534 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 4 | 0.4492 | 2.1145 | 0.3555 | 2.2873 | nan |
| 23/02 | 24 | 4 | 0.3911 | 1.4818 | -0.3837 | 2.2338 | D02M |
| 24/02 | 24 | 4 | 0.4269 | 1.5317 | -0.2485 | 2.2607 | D03M |
| 25/02 | 24 | 1 | 0.0581 | 1.1034 | -0.4160 | 1.3529 | D04M |
| 26/02 | 24 | 3 | 0.2136 | 1.3069 | -1.0364 | 2.1032 | D05M |
| 27/02 | 24 | 3 | 0.3108 | 1.0210 | -0.3840 | 1.4467 | D06M |
| 02/03 | 24 | 1 | 0.3964 | 1.4369 | -0.3220 | 2.1265 | D09M |
| 03/03 | 24 | 1 | 0.1349 | 1.5431 | -0.9265 | 2.3027 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
