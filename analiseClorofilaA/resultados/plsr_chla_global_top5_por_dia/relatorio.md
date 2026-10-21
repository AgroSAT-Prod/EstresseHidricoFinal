# PLSR — CHL A global e predição por data

Pré-processamento: correção de emendas, recorte 400--2450 nm, Savitzky--Golay (11; 2) e SNV. O ranking foi definido pelo módulo do coeficiente PLSR padronizado no modelo global. Antes do Top 5, uma candidata foi mantida somente quando |r de Pearson| < 0.80 com as já selecionadas.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_chl_a | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 133 | 1725 | -0.1143 | 0.1143 | -6.9298 | -0.2972 | nan | True |
| 2 | 10 | 97 | 1365 | 0.0922 | 0.0922 | 9.6950 | 0.1931 | 0.3657 | True |
| 3 | 11 | 83 | 1233 | 0.0919 | 0.0919 | 5.0924 | 0.3038 | 0.3821 | True |
| 4 | 28 | 150 | 1902 | -0.0707 | 0.0707 | -2.1824 | 0.0942 | 0.5446 | True |
| 5 | 29 | 116 | 1564 | 0.0678 | 0.0678 | 3.2283 | -0.1237 | 0.7523 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 168 | 5 | 0.4047 | 1.1791 | 0.3515 | 1.2306 | nan |
| 23/02 | 24 | 5 | 0.5965 | 0.7157 | 0.5037 | 0.7937 | D02M |
| 24/02 | 24 | 2 | 0.5046 | 0.8963 | 0.2877 | 1.0748 | D03M |
| 25/02 | 24 | 5 | 0.7308 | 0.6337 | 0.3884 | 0.9552 | D04M |
| 26/02 | 24 | 3 | 0.6736 | 0.7196 | 0.4628 | 0.9232 | D05M |
| 27/02 | 24 | 3 | 0.4399 | 0.7750 | 0.0507 | 1.0089 | D06M |
| 02/03 | 24 | 5 | 0.4031 | 1.0200 | -0.1027 | 1.3864 | D09M |
| 03/03 | 24 | 1 | 0.2353 | 1.5026 | -0.0017 | 1.7198 | D10M |

As predições por data usam exclusivamente as cinco bandas globais; pontos irrigados são azuis e não irrigados, vermelhos. As métricas CV são diagnósticas, pois a seleção do Top 5 precede a validação cruzada.
