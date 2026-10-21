# CRA — PLSR/VIP por data

Cada uma das 3 datas foi analisada de forma independente. Antes do ajuste, os espectros receberam correção de emendas, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. Bandas localmente redundantes foram agrupadas por Spearman |ρ| > 0,80 em janela <10 nm, mantendo-se a mais associada a CRA em cada grupo.

O PLSR inicial de cada data definiu o VIP. As cinco bandas foram escolhidas em ordem de VIP, mas uma candidata só foi aceita se |r de Pearson| <0,80 contra todas as já selecionadas. O PLSR final e as predições CV por bloco usam somente essas cinco bandas.

## Métricas por data

| data_chl | coleta_espectral | n | representantes_spearman | componentes_top5 | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco |
|---|---|---|---|---|---|---|---|---|
| 24/02 | D03M | 24 | 206 | 4 | 0.5441 | 6.1615 | 0.3342 | 7.4464 |
| 02/03 | D09M | 23 | 206 | 4 | 0.8590 | 4.6579 | 0.7815 | 5.7977 |
| 03/03 | D10M | 24 | 206 | 5 | 0.4777 | 2.0210 | 0.1360 | 2.5994 |

O gráfico `bandas_top5_vip_por_dia.png` concentra as 15 bandas selecionadas. A cor indica o VIP recalculado no modelo final de cada data.
