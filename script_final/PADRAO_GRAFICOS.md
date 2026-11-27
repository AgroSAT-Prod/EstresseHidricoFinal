# Padrão de fontes para gráficos

Configuração de referência: `estilo_graficos.py`. Os tamanhos abaixo são em pontos (pt).

| Elemento | Tamanho | Estilo |
| --- | ---: | --- |
| Títulos de eixos em gráficos amplos | 15 | Negrito |
| Títulos de eixos em painéis compactos | 14 | Negrito |
| Títulos de painéis, quando presentes | 16 | Negrito |
| Números e categorias dos eixos | 13 | Normal |
| Legenda | 13 | Normal |
| Valores acima das barras | 13 | Normal |
| Valores dentro das matrizes | 15 | Normal |

Use os rótulos em inglês. Em figuras com vários painéis ou grupos por dia, identifique-os da esquerda para a direita com `(a)`, `(b)`, etc. Mantenha a legenda dentro da área do gráfico e reserve espaço vertical para ela não colidir com os valores. Não coloque título geral quando ele for desnecessário. Use a fonte padrão do Matplotlib; não force Times New Roman.

No gráfico de métricas FWHM atual, a legenda fica no topo da área dos eixos e o eixo vertical vai de 0 a 1,20. As matrizes FWHM usam títulos como `(a) D02`, sem `n=`.
