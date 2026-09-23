# PLSR — CRA com bandas fixas

Pré-processamento: correção das emendas, interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. As três datas de CRA foram combinadas; a observação sem CRA em 02/03 foi excluída. Validação KFold=4 aleatório, `shuffle=True`, semente 42. O número de componentes foi escolhido pelo maior R² CV.

| Métrica | Resultado |
|---|---:|
| Amostras | 71 |
| Bandas | 400, 531, 570, 739 nm |
| Componentes (maior R² CV) | 2 |
| R² ajuste | 0.500 |
| RMSE ajuste | 12.520 |
| **R² KFold=4** | **0.471** |
| **RMSE KFold=4** | **12.873** |
