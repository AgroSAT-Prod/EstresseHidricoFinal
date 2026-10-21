# CHL Total — PLSR/VIP por data

Cada uma das sete datas foi analisada de forma independente. Antes do ajuste, os espectros receberam correção de emendas, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. Bandas localmente redundantes foram agrupadas por Spearman |ρ| > 0,80 em janela <10 nm, mantendo-se a mais associada ao CHL Total em cada grupo.

O PLSR inicial de cada data definiu o VIP. As cinco bandas foram escolhidas em ordem de VIP, mas uma candidata só foi aceita se |r de Pearson| <0,80 contra todas as já selecionadas. O PLSR final e as predições CV por bloco usam somente essas cinco bandas.

## Métricas por data

| data_chl | coleta_espectral | n | representantes_spearman | componentes_top5 | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco |
|---|---|---|---|---|---|---|---|---|
| 23/02 | D02M | 24 | 206 | 2 | 0.5705 | 1.4950 | 0.4399 | 1.7072 |
| 24/02 | D03M | 24 | 206 | 1 | 0.5543 | 1.6013 | 0.4989 | 1.6978 |
| 25/02 | D04M | 24 | 206 | 2 | 0.6765 | 1.3235 | 0.4520 | 1.7225 |
| 26/02 | D05M | 24 | 207 | 4 | 0.5971 | 1.5144 | 0.3742 | 1.8874 |
| 27/02 | D06M | 24 | 206 | 1 | 0.2975 | 1.7865 | 0.0436 | 2.0845 |
| 02/03 | D09M | 24 | 206 | 2 | 0.4329 | 1.9668 | 0.1482 | 2.4106 |
| 03/03 | D10M | 24 | 206 | 1 | 0.4831 | 1.8878 | 0.3554 | 2.1080 |

O gráfico `bandas_top5_vip_por_dia.png` concentra as 35 bandas selecionadas. A cor indica o VIP recalculado no modelo final de cada data.
