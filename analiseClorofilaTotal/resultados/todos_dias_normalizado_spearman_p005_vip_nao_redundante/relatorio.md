# PLSR — CHL Total, sete datas, espectro normalizado

Pré-processamento: correção das emendas, limpeza/interpolação, recorte 400–2450 nm, Savitzky–Golay e SNV. As sete coletas foram reunidas em um modelo (168 observações).

Redução: Spearman bilateral **p < 0,05** em bandas contíguas, janela <10 nm. Após VIP ≥1, aplicou-se filtro guloso de não redundância: uma banda só permanece se sua correlação Spearman com todas as bandas VIP de maior importância tiver p ≥0,05. A validação é GroupKFold por bloco (quatro dobras).

| Métrica | Resultado |
|---|---:|
| Representantes Spearman p <0,05 | 206 |
| Candidatas VIP ≥1 | 60 |
| Bandas após filtro | 2 |
| Componentes PLSR | 1 |
| R² ajuste | 0.334 |
| RMSE ajuste | 2.152 |
| **R² CV por bloco** | **0.315** |
| **RMSE CV por bloco** | **2.184** |
