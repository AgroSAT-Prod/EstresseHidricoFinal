# Parâmetros fisiológicos × índices espectrais

Experimento independente que adapta o procedimento de
[`relacaoCRA`](../relacaoCRA/analisar_indices_cra.py) para os nove parâmetros
medidos em todos os sete dias: **CHL A, CHL B, CHL TOTAL, Tleaf, A, Ci, gs, E e VPD**.

## Executar

Na raiz do repositório, com as dependências de `requirements.txt` instaladas:

```bash
python3 parametros_vs_indices/analisar_parametros_vs_indices.py
```

O script usa um backend sem interface gráfica e grava todos os resultados em
`parametros_vs_indices/resultados/`. Uma nova execução substitui os arquivos de
mesmo nome. Os arquivos de entrada e o experimento CRA são somente lidos.

Testes (usa `unittest`, da biblioteca padrão):

```bash
python3 -m unittest discover -s parametros_vs_indices/tests -v
```

Os testes incluem comparação dos cálculos com a função original do CRA e uma
verificação de cobertura com os arquivos reais do repositório.

## Dados e unidade experimental

- Fisiologia: `ParametrosFisiologicos.xlsx`, aba inicial.
- Espectros: `dataset/Unificada13052026_Limpa.csv`, somente turno da manhã.
- Parcela: bloco × genótipo × condição. Cada data é pareada separadamente.
- As quatro linhas de cada genótipo × condição são associadas, na ordem da
  planilha, a B1–B4, antes de excluir ausências. **Essa é uma suposição herdada
  do CRA**, pois a planilha não contém uma coluna explícita de bloco.
- A coluna `Data` da planilha não determina a data de cada medida; usam-se as
  datas nos cabeçalhos, aceitando ponto ou barra e espaços redundantes.

| Data fisiológica | Coleta espectral |
|---|---|
| 23/02 | D02M |
| 24/02 | D03M |
| 25/02 | D04M |
| 26/02 | D05M |
| 27/02 | D06M |
| 02/03 | D09M |
| 03/03 | D10M |

As reflectâncias brutas são reduzidas à **média por parcela e coleta antes do
cálculo dos índices**. Não se aplica suavização ou SNV. Todas as leituras
disponíveis entram na média; as contagens são exportadas. Nos dados atuais,
167 parcelas-data possuem oito leituras e CD202/IRRIG/B1 em D09M possui 12.
Cada banda usa suas leituras numéricas finitas disponíveis, e a auditoria
registra quantas foram utilizadas.

| Índice | Fórmula implementada |
|---|---|
| PRI | `(R531 − R570) / (R531 + R570)` |
| NDVI | `(R800 − R670) / (R800 + R670)` |
| NDRE | `(R790 − R720) / (R790 + R720)` |
| WDI | `R970 / R900` |
| MSI | `R1600 / R820` |

As fórmulas seguem os scripts existentes em `indicesEspectrais`. **WDI** mantém
o nome e a convenção dessa pasta; sua fórmula é a inversa do WBI utilizado no CRA.

## Estatística

Para cada parâmetro × índice × data, reúnem-se os três genótipos e as duas
condições hídricas: atualmente **24 parcelas por comparação**. Usam-se as
mesmas chamadas do CRA: `pearsonr`, `spearmanr` (p aproximado) e `linregress`.
São exportados **somente p-valores brutos**, sem ajuste múltiplo. A referência
`p ≤ 0,05` é exploratória diante do número de comparações. O R² descreve o
ajuste nos próprios dados e não mede desempenho em validação preditiva.

Há 45 comparações por dia, **315 nas sete datas**, além de 45 comparações
reunindo todas as datas: **360 linhas estatísticas**. O agrupamento temporal
é **apenas descritivo**: contém observações repetidas das mesmas parcelas e
pode refletir a mudança simultânea das variáveis no tempo. As relações por
data também podem refletir diferenças entre genótipos e condições; não
constituem evidência causal ou uma comparação estatística entre tratamentos.

Ausências, valores não numéricos e infinitos são removidos somente do par
parâmetro–índice afetado. O cálculo é interrompido se houver menos de oito
pares válidos em qualquer comparação, identificando parâmetro, índice e data
na mensagem de erro. Se uma variável for constante, o grupo recebe status
explicativo e métricas ausentes. Colunas obrigatórias
ausentes, chaves duplicadas no pareamento e estrutura diferente de quatro
repetições por genótipo × condição causam erro contextualizado.

## Resultados

Comece pelo [relatório completo](resultados/relatorio.md).

| Arquivo ou pasta em `resultados/` | Conteúdo |
|---|---|
| `dados_pareados.csv` | Uma linha por parcela-data e parâmetro: chaves, linha original, valor fisiológico, contagem de leituras e cinco índices. Ausências são preservadas para auditoria. |
| `medias_espectrais_indices.csv` | Uma linha por parcela-data: dez reflectâncias médias, contagens totais e válidas por banda, cinco índices. |
| `auditoria_pareamento.csv` | Vínculo da linha da planilha com bloco, genótipo, condição e coleta; situação de pareamento e contagens de leituras. |
| `correlacoes.csv` | Parâmetro, data, tipo de análise, índice, fórmula, n válido/total/excluído, status, Pearson e p, Spearman e p, R², inclinação e intercepto. |
| `agrupados/<parametro>/` | 45 figuras, uma por parâmetro × índice; sete painéis de datas com todos os genótipos no mesmo eixo. Exemplo: `agrupados/chl_a/pri.png`. |
| `heatmaps/` | Nove mapas de Spearman; cinco índices × sete datas e uma coluna temporal descritiva. |
| `relatorio.md` | Método, cobertura, fórmulas, limitações, tabelas e links das figuras. |

CSVs usam separador `;`, ponto decimal e até 12 algarismos significativos.
Todas as **54 figuras** (45 figuras de dispersão e nove mapas de calor) são
salvas em PNG e PDF, totalizando 108 arquivos de imagem/documento. Os formatos
anteriores são removidos na execução: figuras com várias datas em um painel,
a antiga pasta `separados/`, imagens com painéis separados por genótipo e
imagens por data com painéis de índices.
A remoção se limita aos nomes de arquivos produzidos pelo experimento.

Cada imagem fixa um índice para determinado parâmetro e reúne **sete painéis
de datas**, em ordem cronológica e com escalas compartilhadas. **BR16, CD202 e
EMB48 ficam juntos no mesmo eixo**, todos com marcadores circulares. A cor
identifica irrigados e não irrigados. Cada ponto é uma parcela identificada
por bloco. Os gráficos mostram somente pontos, sem linhas de tendência.
Os coeficientes da regressão linear e o R² permanecem nas tabelas estatísticas,
assim como as correlações e p-valores, reunindo todos os genótipos e ambas as
condições, com 24 parcelas por data.
