# Análise do parâmetro E

Execute `python analiseE/analisar_e.py` a partir da raiz do projeto (ou use o caminho absoluto do script). O programa lê as funções diretamente de `script_final/Pipeline_Final_Predicao.ipynb` e aplica a E o mesmo pré-processamento, agrupamento Spearman, seleção Top 5 VIP e PLSR diário usado nos demais parâmetros. O notebook original não é alterado.

Arquivos gerados nesta pasta:

- `metricas_e_por_dia.csv`: tamanho da amostra, número de representantes, cinco bandas, componentes, R² e RMSE de ajuste e de validação por bloco.
- `bandas_e_top5_vip.csv`: VIP de seleção, VIP do modelo final e correlação Spearman da banda com E.
- `predicoes_e_cv_bloco.csv`: observado, predito e resíduo por parcela.
- `e/top5_vip_DD_MM.csv`, `e/componentes_iniciais_DD_MM.csv` e `e/componentes_finais_DD_MM.csv`: detalhes por data.

A validação deixa um bloco de fora, mas a escolha das bandas e dos componentes usa todos os dados da data antes das dobras. Por isso, R² e RMSE de CV são diagnósticos e podem ser otimistas.
