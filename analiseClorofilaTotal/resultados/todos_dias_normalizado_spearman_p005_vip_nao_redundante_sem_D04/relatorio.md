# PLSR — CHL Total, 6 datas, espectro normalizado

Pré-processamento: correção das emendas, limpeza/interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. As coletas 23/02, 24/02, 26/02, 27/02, 02/03, 03/03 foram reunidas em um modelo (144 observações). A coleta D04M foi excluída.

Redução: Spearman bilateral **p < 0,05** em bandas contíguas, janela <10 nm. Após VIP ≥1, aplicou-se filtro guloso de não redundância: uma banda só permanece se sua correlação Spearman com todas as bandas VIP de maior importância tiver p ≥0,05. A validação é GroupKFold por bloco (quatro dobras).

| Métrica | Resultado |
|---|---:|
| Representantes Spearman p <0,05 | 206 |
| Candidatas VIP ≥1 | 60 |
| Bandas após filtro | 3 |
| Componentes PLSR | 2 |
| R² ajuste | 0.333 |
| RMSE ajuste | 2.192 |
| **R² CV por bloco** | **0.297** |
| **RMSE CV por bloco** | **2.251** |
