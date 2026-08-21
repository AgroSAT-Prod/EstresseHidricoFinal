# Random Forest — bandas VIP consolidadas por genótipo

Cada modelo usa somente as cinco bandas Top VIP do respectivo genótipo, obtidas com os sete dias reunidos. A classificação é IRRIG × NIRRIG usando todas as observações da manhã de D02, D03, D04, D05, D06, D09 e D10.

Validação: StratifiedGroupKFold (k=5; semente 42), com `nomenclaura` como grupo. Assim, as oito leituras de um mesmo arquivo permanecem inteiramente no treino ou no teste.

| Genótipo | Bandas (nm) | Acc ± DP | F1 ± DP | κ ± DP | AUC-ROC ± DP |
|---|---|---:|---:|---:|---:|
| BR16 | 1963, 1953, 1951, 405, 1943 | 0.893 ± 0.020 | 0.893 ± 0.026 | 0.785 ± 0.040 | 0.958 ± 0.004 |
| CD202 | 1480, 1997, 2007, 1987, 1470 | 0.950 ± 0.019 | 0.948 ± 0.021 | 0.899 ± 0.038 | 0.988 ± 0.014 |
| EMB48 | 1943, 1939, 1938, 1948, 1933 | 0.850 ± 0.020 | 0.845 ± 0.045 | 0.690 ± 0.043 | 0.938 ± 0.012 |
