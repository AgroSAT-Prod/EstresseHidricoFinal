# PLSR — CHL Total com bandas fixas, sem D04M

Pré-processamento: correção das emendas, interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. Seis datas combinadas, excluindo D04M; GroupKFold por bloco (k=4).

| Métrica | Resultado |
|---|---:|
| Amostras | 144 |
| Bandas | 400, 531, 570, 739 nm |
| Componentes PLSR | 4 |
| R² ajuste | 0.372 |
| RMSE ajuste | 2.128 |
| **R² CV por bloco** | **0.322** |
| **RMSE CV por bloco** | **2.211** |
