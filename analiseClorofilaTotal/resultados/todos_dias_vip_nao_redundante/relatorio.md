# PLSR — Clorofila Total, todas as sete coletas

As sete datas foram combinadas num único modelo (168 observações; D02–D06, D09 e D10). As leituras foram agregadas por bloco × genótipo × condição. Aplicou-se Spearman |ρ| ≥ 0,80 entre bandas contíguas, selecionando uma representante por grupo pela maior associação absoluta com CHL Total; o PLSR escalonado reteve VIP ≥ 1,0. Ao final, aplicou-se filtro guloso: uma banda só permanece se tiver |ρ| < 0,80 com todas as bandas VIP de maior importância já mantidas.

A validação foi GroupKFold com quatro dobras: cada dobra exclui um bloco inteiro, em todas as datas. O número de componentes foi escolhido pelo menor RMSE dessa CV.

| Métrica | Resultado |
|---|---:|
| Amostras válidas | 168 |
| Representantes após Spearman | 216 |
| Bandas após filtro de não redundância | 6 |
| Componentes PLSR | 5 |
| R² de ajuste | 0.373 |
| RMSE de ajuste | 2.088 |
| **R² CV por bloco** | **0.331** |
| **RMSE CV por bloco** | **2.157** |

O R²/RMSE CV é diagnóstico porque a seleção Spearman/VIP foi feita antes das dobras; uma estimativa estritamente imparcial exigiria seleção aninhada em cada dobra.
