# Condutância estomática (gs) — PLSR/VIP por data

Cada uma das sete datas foi analisada de forma independente. Antes do ajuste, os espectros receberam correção de emendas, recorte 400--2450 nm, Savitzky--Golay (janela 11; polinômio 2) e SNV. Bandas localmente redundantes foram agrupadas por Spearman |ρ| > 0,80 em janela <10 nm, mantendo-se a mais associada a Condutância estomática (gs) em cada grupo.

O PLSR inicial de cada data definiu o VIP. As cinco bandas foram escolhidas em ordem de VIP, mas uma candidata só foi aceita se |r de Pearson| <0,80 contra todas as já selecionadas. O PLSR final e as predições CV por bloco usam somente essas cinco bandas.

## Métricas por data

| data_chl | coleta_espectral | n | representantes_spearman | componentes_top5 | r2_ajuste | rmse_ajuste | r2_cv_bloco | rmse_cv_bloco |
|---|---|---|---|---|---|---|---|---|
| 23/02 | D02M | 24 | 206 | 5 | 0.4785 | 0.1348 | 0.0386 | 0.1830 |
| 24/02 | D03M | 24 | 206 | 2 | 0.7035 | 0.1521 | 0.5897 | 0.1789 |
| 25/02 | D04M | 24 | 206 | 4 | 0.7736 | 0.0779 | 0.5469 | 0.1102 |
| 26/02 | D05M | 24 | 207 | 4 | 0.7289 | 0.1345 | 0.6407 | 0.1549 |
| 27/02 | D06M | 24 | 206 | 5 | 0.8902 | 0.0881 | 0.8273 | 0.1104 |
| 02/03 | D09M | 24 | 206 | 3 | 0.7600 | 0.1505 | 0.6150 | 0.1906 |
| 03/03 | D10M | 24 | 206 | 1 | 0.6654 | 0.1530 | 0.5040 | 0.1863 |

O gráfico `bandas_top5_vip_por_dia.png` concentra as 35 bandas selecionadas. A cor indica o VIP recalculado no modelo final de cada data.
