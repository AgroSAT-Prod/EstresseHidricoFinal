# Resumo da Base de Dados Unificada

Este arquivo resume, de forma simples, a quantidade de amostras e as principais caracteristicas do dataset de leituras hiperspectrais.

## Visao geral

O dataset esta organizado com os metadados no inicio e, em seguida, as colunas numeradas que representam as bandas hiperspectrais. As leituras cobrem uma faixa ampla do espectro, de **350 nm a 2500 nm**, com uma banda por nanometro.

- Total de amostras: **1.924**
- Total de colunas: **2.157**
- Colunas de metadados: **6**
- Bandas hiperspectrais: **2.151**
- Intervalo das bandas: **350 a 2500 nm**
- Passo entre bandas: **1 nm**
- Valores ausentes: **0**
- Linhas duplicadas: **0**

## Metadados

As colunas de metadados identificadas antes das bandas numeradas sao:

- `nomenclaura`
- `bloco`
- `genotipo`
- `condicao`
- `data_coleta`
- `turno`

## Quantidade de amostras por metadado

### Condicao

| Condicao | Amostras |
|---|---:|
| IRRIG | 980 |
| NIRRIG | 944 |

### Genotipo

| Genotipo | Amostras |
|---|---:|
| CD202 | 644 |
| BR16 | 640 |
| EMB48 | 640 |

### Bloco

| Bloco | Amostras |
|---|---:|
| B1 | 484 |
| B2 | 480 |
| B3 | 480 |
| B4 | 480 |

### Data de coleta

| Data de coleta | Amostras |
|---|---:|
| D02M | 192 |
| D02T | 192 |
| D03M | 192 |
| D03T | 192 |
| D04M | 192 |
| D05M | 192 |
| D06M | 192 |
| D09M | 196 |
| D09T | 192 |
| D10M | 192 |

### Turno

| Turno | Amostras |
|---|---:|
| manha | 1348 |
| tarde | 576 |

### Nomenclaura

- Valores unicos: **408**
- Este campo identifica os arquivos/amostras originais, por exemplo `B1_BR16_IRRIG_REPROD00000.asd`.
