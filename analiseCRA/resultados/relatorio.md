# PLSR — CRA nas três coletas

Pareamento: média de leituras espectrais matinais por bloco × genótipo × condição; D03M=24/02, D09M=02/03 e D10M=03/03.

Spearman |ρ| ≥ 0,80 entre bandas contíguas (janela <10 nm); uma representante por grupo, escolhida pelo maior |ρ| com CRA. O PLSR é escalonado; componentes pelo menor RMSE em CV deixando um bloco de fora. VIP ≥1,0 do ajuste inicial é retido e o modelo final é reajustado.

`R² CV-bloco` é diagnóstico: a seleção Spearman/VIP ocorreu antes das dobras e pode tornar a estimativa otimista.

| CRA | Coleta espectral | n | Rep. Spearman | VIP ≥1 | Componentes | R² ajuste | R² CV-bloco | RMSE CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 24/02 | D03M | 24 | 216 | 73 | 3 | 0.552 | 0.374 | 7.2176 |
| 02/03 | D09M | 23 | 217 | 79 | 1 | 0.822 | 0.771 | 5.9344 |
| 03/03 | D10M | 24 | 216 | 104 | 2 | 0.224 | -0.137 | 2.9823 |
