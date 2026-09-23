# iPLS/PLSR — ETR

Pareamento: média das leituras espectrais matinais por bloco × genótipo × condição; D02M=23/02, D09M=02/03 e D10M=03/03.

Em cada data, calcula-se Spearman bilateral entre cada banda e ETR; somente p < 0,05 entra. As bandas aprovadas são separadas em intervalos fixos de 10 nm (iPLS), dos quais se mantém a maior |rho|. O PLSR escalonado escolhe os componentes pelo menor RMSE em validação cruzada deixando um bloco de fora; VIP ≥ 1,0 é retido no modelo final.

Quando nenhuma banda atende p < 0,05, o PLSR/VIP não é ajustado — não há substituição do limiar. As métricas CV são diagnósticas: Spearman/iPLS/VIP foram definidos antes das dobras e, portanto, ainda podem ser otimistas. Os gráficos Top 5 mostram regressões lineares univariadas reflectância × ETR; o CSV contém métricas de ajuste e CV por bloco.

| ETR | Coleta espectral | n | Bandas Spearman p < 0,05 | Intervalos iPLS | VIP ≥1 | Componentes | R² ajuste | RMSE ajuste | R² CV-bloco | RMSE CV-bloco | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 23/02 | D02M | 24 | 3 | 2 | 1 | 1.000 | 0.157 | 31.338 | -0.287 | 38.724 | Modelo ajustado |
| 02/03 | D09M | 24 | 2066 | 209 | 111 | 1.000 | 0.519 | 21.628 | 0.486 | 22.379 | Modelo ajustado |
| 03/03 | D10M | 24 | 0 | 0 | 0 | — | — | — | — | — | Sem bandas com Spearman p < 0.05; PLSR/VIP não ajustado. |
