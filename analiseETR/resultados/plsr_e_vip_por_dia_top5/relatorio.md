# Evapotranspiração (E) — PLSR/VIP por data

Cada uma das sete datas foi analisada de forma independente. Antes do ajuste, os espectros receberam correção de emendas, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. Bandas localmente redundantes foram agrupadas por Spearman |ρ| > 0,80 em janela <10 nm, mantendo-se a mais associada a Evapotranspiração (E) em cada grupo.

O PLSR inicial de cada data definiu o VIP. As cinco bandas foram escolhidas em ordem de VIP, mas uma candidata só foi aceita se |r de Pearson| <0,80 contra todas as já selecionadas. O PLSR final e as predições CV por bloco usam somente essas cinco bandas.

## Métricas por data

| data_chl | coleta_espectral | n | representantes_spearman | componentes_top5 | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco |
|---|---|---|---|---|---|---|---|---|
| 23/02 | D02M | 24 | 206 | 1 | 0.2652 | 1.1344 | -0.0935 | 1.3839 |
| 24/02 | D03M | 24 | 206 | 2 | 0.8079 | 1.1994 | 0.7286 | 1.4257 |
| 25/02 | D04M | 24 | 206 | 1 | 0.5562 | 1.2655 | 0.3842 | 1.4908 |
| 26/02 | D05M | 24 | 207 | 1 | 0.6936 | 1.9337 | 0.6142 | 2.1698 |
| 27/02 | D06M | 24 | 206 | 1 | 0.8162 | 0.9074 | 0.7315 | 1.0968 |
| 02/03 | D09M | 24 | 206 | 3 | 0.7978 | 1.3976 | 0.6402 | 1.8644 |
| 03/03 | D10M | 24 | 206 | 1 | 0.6614 | 1.2548 | 0.5158 | 1.5005 |

O gráfico `bandas_top5_vip_por_dia.png` concentra as 35 bandas selecionadas. A cor indica o VIP recalculado no modelo final de cada data.
