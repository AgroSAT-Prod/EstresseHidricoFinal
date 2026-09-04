# PLS-DA — VIP entre as bandas selecionadas nas runs diárias

Para cada genótipo, as bandas candidatas são a união das Top 5 VIP das sete runs por dia (D02, D03, D04, D05, D06, D09 e D10). O modelo reúne todos os sete dias, mantendo os genótipos separados e classificando IRRIG × NIRRIG.

A validação usa StratifiedKFold (k=4), sem GroupKFold, repetida nas sementes 42–46 (20 dobras por genótipo). Em cada dobra, as cinco bandas VIP são escolhidas somente no treino. O ranking exibido é o VIP descritivo do ajuste com todas as observações.

| Genótipo | Bandas candidatas | Top 5 (nm; VIP final) | Acc (IC95%) | F1 (IC95%) | κ (IC95%) | AUC-ROC (IC95%) |
|---|---:|---|---:|---:|---:|---:|
| BR16 | 27 | 1963 (1.303), 1953 (1.301), 1951 (1.298), 405 (1.282), 1943 (1.278) | 0.790 [0.773; 0.806] | 0.806 [0.792; 0.821] | 0.579 [0.546; 0.613] | 0.894 [0.882; 0.907] |
| CD202 | 35 | 1480 (1.274), 1997 (1.274), 2007 (1.273), 1987 (1.273), 1470 (1.241) | 0.837 [0.820; 0.854] | 0.821 [0.800; 0.841] | 0.674 [0.640; 0.708] | 0.942 [0.932; 0.952] |
| EMB48 | 31 | 1943 (1.253), 1939 (1.253), 1938 (1.252), 1948 (1.247), 1933 (1.243) | 0.829 [0.813; 0.845] | 0.835 [0.821; 0.849] | 0.658 [0.626; 0.690] | 0.912 [0.899; 0.925] |
