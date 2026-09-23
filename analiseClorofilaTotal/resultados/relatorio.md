# PLSR — Clorofila Total nas sete coletas

Pareamento: média de leituras espectrais matinais por bloco × genótipo × condição; D02M–D06M=23–27/02, D09M=02/03 e D10M=03/03.

Spearman |ρ| ≥ 0,80 entre bandas contíguas (janela <10 nm); uma representante por grupo, escolhida pelo maior |ρ| com Clorofila Total. O PLSR é escalonado, com componentes escolhidos pelo menor RMSE em CV deixando um bloco de fora. VIP ≥1,0 do ajuste inicial é retido e o modelo final é reajustado.

`R² CV-bloco` é diagnóstico: a seleção Spearman/VIP ocorreu antes das dobras e pode tornar a estimativa otimista.

| Clorofila Total | Coleta espectral | n | Rep. Spearman | VIP ≥1 | Componentes | R² ajuste | R² CV-bloco | RMSE CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 23/02 | D02M | 24 | 218 | 105 | 1 | 0.620 | 0.469 | 1.6621 |
| 24/02 | D03M | 24 | 216 | 39 | 6 | 0.768 | 0.149 | 2.2121 |
| 25/02 | D04M | 24 | 216 | 32 | 3 | 0.707 | 0.531 | 1.5942 |
| 26/02 | D05M | 24 | 216 | 61 | 4 | 0.560 | 0.100 | 2.2633 |
| 27/02 | D06M | 24 | 217 | 41 | 6 | 0.686 | 0.181 | 1.9288 |
| 02/03 | D09M | 24 | 217 | 79 | 3 | 0.574 | 0.326 | 2.1436 |
| 03/03 | D10M | 24 | 216 | 30 | 2 | 0.391 | 0.092 | 2.5020 |
