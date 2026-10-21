# PLSR — CRA global e predição por data

Pré-processamento aplicado antes da agregação por bloco: correção de emendas dos detectores, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. O ranking foi calculado no PLSR único das três datas. O coeficiente é padronizado; sinal positivo indica aumento previsto de CRA quando a reflectância da banda aumenta, mantendo as demais bandas constantes. A ordem usa o módulo do coeficiente. Antes do Top 5, excluíram-se bandas multicolineares: uma candidata só foi mantida se |r de Pearson| < 0.80 contra cada banda já selecionada.

## Top 5 global

| ranking_top5_nao_colinear | ranking_coeficiente_plsr | indice_representante | banda_nm | coeficiente_plsr_padronizado | coeficiente_plsr_absoluto | coeficiente_plsr_bruto | rho_spearman_cra | max_abs_r_pearson_com_banda_ja_selecionada | selecionada_top5_nao_colinear |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 96 | 1354 | 0.0122 | 0.0122 | 14.6151 | 0.5511 | nan | True |
| 2 | 2 | 127 | 1668 | 0.0118 | 0.0118 | 9.0480 | 0.0964 | 0.2155 | True |
| 3 | 12 | 90 | 1296 | 0.0090 | 0.0090 | 5.4301 | 0.7201 | 0.7808 | True |
| 4 | 69 | 156 | 1962 | -0.0064 | 0.0064 | -2.1892 | -0.5117 | 0.6786 | True |
| 5 | 119 | 33 | 733 | -0.0057 | 0.0057 | -1.9240 | 0.0807 | 0.5378 | True |

## Métricas

| escopo | n | componentes | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco | coleta_espectral |
|---|---|---|---|---|---|---|---|
| Global (todas as datas; Top 5) | 71 | 5 | 0.5593 | 11.7498 | 0.4653 | 12.9412 | nan |
| 24/02 | 24 | 1 | 0.3930 | 7.1097 | 0.2046 | 8.1388 | D03M |
| 02/03 | 23 | 4 | 0.8067 | 5.4523 | 0.6382 | 7.4600 | D09M |
| 03/03 | 24 | 2 | 0.2884 | 2.3590 | -0.1819 | 3.0403 | D10M |

As predições diárias usam somente estas cinco bandas. Componentes foram escolhidos pelo menor RMSE em GroupKFold por bloco (B1--B4). As métricas de CV são diagnósticas: o Top 5 foi selecionado no conjunto global antes da CV.
