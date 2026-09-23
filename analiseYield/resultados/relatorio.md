# PLSR — Yield nas três coletas

Pareamento: média de leituras espectrais matinais por bloco × genótipo × condição; D02M=23/02, D09M=02/03 e D10M=03/03.

Spearman |ρ| ≥ 0,80 foi aplicado entre bandas contíguas (janela <10 nm). Em cada grupo, manteve-se a banda de maior |ρ| com Yield. O PLSR foi escalonado, com componentes escolhidos pelo menor RMSE em validação cruzada deixando um bloco de fora; VIP ≥1,0 do ajuste inicial foi retido e o modelo final reajustado.

`R² CV-bloco` é diagnóstico: a seleção de representantes/VIP foi feita antes das dobras; portanto, a estimativa pode ser otimista e não substitui uma seleção aninhada em nova amostra.

| Yield | Coleta espectral | n | Rep. Spearman | VIP ≥1 | Componentes | R² ajuste | R² CV-bloco | RMSE CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 23/02 | D02M | 24 | 218 | 35 | 2 | 0.381 | 0.136 | 0.0501 |
| 02/03 | D09M | 24 | 217 | 119 | 1 | 0.517 | 0.485 | 0.0520 |
| 03/03 | D10M | 24 | 216 | 49 | 6 | 0.762 | 0.183 | 0.0580 |
