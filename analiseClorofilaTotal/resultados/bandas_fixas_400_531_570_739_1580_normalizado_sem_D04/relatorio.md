# PLSR — CHL Total com bandas fixas, sem D04M

Pré-processamento: correção das emendas, interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. Seis datas combinadas, excluindo D04M; GroupKFold por bloco (k=4).

| Métrica | Resultado |
|---|---:|
| Amostras | 144 |
| Bandas | 400, 531, 570, 739, 1580 nm |
| Componentes PLSR | 5 |
| R² ajuste | 0.374 |
| RMSE ajuste | 2.123 |
| **R² CV por bloco** | **0.321** |
| **RMSE CV por bloco** | **2.212** |
