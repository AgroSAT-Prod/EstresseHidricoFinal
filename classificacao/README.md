# Experimento de Classificação

Esta pasta contém o experimento de classificação para distinguir a condição hídrica das amostras:

- `IRRIG`
- `NIRRIG`

O alvo usado pelos modelos é a coluna `condicao`, com `NIRRIG` como classe positiva para as métricas binárias e AUC-ROC.

## Objetivo

Treinar e avaliar modelos de classificação usando apenas bandas hiperspectrais previamente selecionadas pelos testes estatísticos do projeto. A seleção das bandas não é refeita nesta pasta; os scripts apenas leem os CSVs de bandas já existentes e treinam os modelos com elas.

## Dataset de entrada

A classificação usa os CSVs espectrais já pré-processados na pasta `dataset/`. O arquivo usado depende da opção `--estagio`:

- `recortado`: `dataset/Unificada13052026_400_2450.csv`
- `suavizado`: `dataset/Unificada13052026_suavizado.csv`
- `normalizado`: `dataset/Unificada13052026_normalizado.csv`

O estágio padrão é `normalizado`. Portanto, por padrão, os modelos usam `dataset/Unificada13052026_normalizado.csv` como fonte dos valores das bandas.

## Modos de bandas

A flag `--modo` define quais bandas entram no experimento:

- `agrupado`: usa as Top 5 bandas gerais de `selecaoVariaveis/dataset_gerado/top5_bandas.csv`.
- `por_genotipo`: executa um cenário separado para cada genótipo (`BR16`, `CD202`, `EMB48`) usando `dataset/dataset_gerado/bandas_<genotipo>.csv`.
- `ambos`: executa os modos `agrupado` e `por_genotipo`.

Quando um CSV por genótipo possui mais de 5 bandas, apenas as 5 primeiras pelo `rank` são usadas.

## Avaliação

A avaliação usa `StratifiedGroupKFold` com:

- `k = 5`
- `shuffle = True`
- `random_state = 42`
- grupo de validação: `nomenclaura`

Essa estratégia preserva a estratificação da classe e impede que leituras com a mesma `nomenclaura` apareçam simultaneamente no treino e no teste do mesmo fold.

## Modelos

Os modelos avaliados são:

- Random Forest
- Support Vector Machine
- PLS-DA

As métricas avaliadas por fold e resumidas em média/desvio são:

- accuracy
- precision
- recall
- f1-score
- kappa
- AUC-ROC

## Estrutura da pasta

- `classificacao.py`: script principal; executa todos os modelos.
- `visualizacao_resultados_classificacao.ipynb`: notebook para visualizar tabelas e gráficos a partir dos CSVs salvos.
- `scripts/`: scripts auxiliares e scripts individuais por modelo.
- `outputs/`: CSVs gerados pela execução dos experimentos.

## Como executar

Executar todos os modelos com as bandas gerais:

```bash
python classificacao.py --modo agrupado
```

Executar todos os modelos por genótipo:

```bash
python classificacao.py --modo por_genotipo
```

Executar todos os cenários:

```bash
python classificacao.py --modo ambos
```

Executar sem salvar CSVs:

```bash
python classificacao.py --modo ambos --sem-csv
```

Executar um modelo específico:

```bash
python scripts/random_forest_top5.py --modo agrupado
python scripts/svm_top5.py --modo por_genotipo
python scripts/pls_da_top5.py --modo ambos
```

Escolher o estágio de preprocessamento:

```bash
python classificacao.py --modo agrupado --estagio normalizado
```

Opções aceitas em `--estagio`:

- `recortado`
- `suavizado`
- `normalizado`

## Outputs

Os resultados são salvos em:

- `outputs/agrupado/<modelo>/`
- `outputs/por_genotipo/<genotipo>/<modelo>/`

Cada pasta de modelo/cenário contém:

- `configuracao.csv`: configuração do experimento, bandas usadas, alvo, classe positiva, validador e grupo de validação.
- `fold_assignments.csv`: amostras de treino e teste em cada fold.
- `metricas_folds.csv`: métricas calculadas em cada fold.
- `metricas_resumo.csv`: média e desvio das métricas nos 5 folds.
- `predicoes.csv`: predições out-of-fold com classe real, classe predita, score e valores das bandas usadas.
- `matriz_confusao_folds.csv`: matriz de confusão por fold em formato longo.
- `relatorio_classificacao_folds.csv`: precision, recall, f1-score e support por classe em cada fold.
- `curva_roc_folds.csv`: pontos da curva ROC por fold.

Outputs específicos por modelo:

- Random Forest: `importancia_bandas.csv`
- SVM: `importancia_permutacao.csv` e `parametros_modelo.csv`
- PLS-DA: `coeficientes_pls.csv`, `pesos_pls.csv` e `parametros_modelo.csv`

O notebook de visualização usa esses CSVs para gerar tabelas e gráficos sem rerodar os modelos.
