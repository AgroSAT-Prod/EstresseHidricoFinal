"""Gera o PNG dos fluxos descritos nos dois notebooks Pipeline_Final_*.

Execução: python script_final/gerar_fluxograma_unificado.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
W, H = 2600, 2780
im = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(im)
FONT = Path("C:/Windows/Fonts")
INK, MUTED = "#19344a", "#486174"
GREEN, BLUE, PURPLE = "#167361", "#245f9c", "#795092"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT / ("arialbd.ttf" if bold else "arial.ttf")), size)


def text(x, y, s, size=31, color=INK, bold=False):
    d.text((x, y), s, font=font(size, bold), fill=color, anchor="mm")


def box(x, y, w, h, title, lines=(), color=GREEN, fill="#eef8f4", title_size=36, body_size=30):
    d.rounded_rectangle((x, y, x+w, y+h), radius=23, fill=fill, outline=color, width=2)
    items = [(title, title_size, True)] + [(s, body_size, False) for s in lines]
    heights = [sz + 14 for _, sz, _ in items]
    cy = y + (h-sum(heights))/2
    for (s, sz, bold), height in zip(items, heights):
        text(x+w/2, cy+height/2, s, sz, color if bold else MUTED, bold)
        cy += height


def arrow(points, color="#658294", width=4):
    d.line(points, fill=color, width=width, joint="curve")
    x, y = points[-1]
    px, py = points[-2]
    if y > py:
        triangle = [(x,y), (x-10,y-17), (x+10,y-17)]
    elif x > px:
        triangle = [(x,y), (x-17,y-10), (x-17,y+10)]
    else:
        triangle = [(x,y), (x+17,y-10), (x+17,y+10)]
    d.polygon(triangle, fill=color)


text(1300, 67, "Fluxograma de predição e classificação", 57, INK, True)
text(1300, 123, "Etapas compartilhadas e caminhos específicos da análise espectral", 32, MUTED)

box(540, 180, 1520, 105, "Dados espectrais + metadados", ["Unificada13052026.csv · bloco, genótipo, condição e data"])
arrow([(1300,285),(1300,325)])
box(540, 325, 1520, 110, "Limpeza dos metadados", ["Conferir genótipo e condição pela identificação; padronizar os blocos"])
arrow([(1300,435),(1300,475)])
box(540, 475, 1520, 185, "Pré-processamento espectral", [
    "Recorte: 400–2450 nm · remoção de valores extremos (|valor| > 10.000)",
    "Correção de emendas em 1000 e 1800 nm · interpolação de ausentes",
    "Savitzky–Golay (janela 11, grau 2) → SNV por espectro",
])
arrow([(1300,660),(1300,700)])
box(540, 700, 1520, 110, "Selecionar coletas da manhã e analisar por dia", ["Três genótipos reunidos em cada data"])
arrow([(1300,810),(1300,855),(660,855),(660,900)], BLUE)
arrow([(1300,855),(1940,855),(1940,900)], PURPLE)

d.rounded_rectangle((90,900,1230,2680), 30, fill="#f5f9fd")
d.rounded_rectangle((1370,900,2510,2680), 30, fill="#faf7fc")
text(660,952,"PREDIÇÃO FISIOLÓGICA",40,BLUE,True)
text(1940,952,"CLASSIFICAÇÃO DA CONDIÇÃO HÍDRICA",36,PURPLE,True)

def pred(y,h,title,lines):
    box(120,y,1080,h,title,lines,BLUE,"#edf4fc",35,29)


pred(1010,195,"Média por parcela + medidas fisiológicas",[
    "Parear ParametrosFisiologicos.xlsx por",
    "bloco × genótipo × condição × data",
    "Excluir ausentes somente do alvo e da data afetados",
])
arrow([(660,1205),(660,1250)],BLUE)
pred(1250,155,"Agrupar bandas vizinhas por Spearman",[
    "|ρ| > 0,80 · ligação completa · extensão < 10 nm",
    "Agrupamento separado para cada alvo e data",
])
arrow([(660,1405),(660,1450)],BLUE)
pred(1450,130,"Escolher uma representante por grupo",[
    "Banda com maior |ρ| em relação ao parâmetro fisiológico",
])
arrow([(660,1580),(660,1625)],BLUE)
pred(1625,155,"PLSR inicial nas bandas representantes",[
    "Até 5 componentes; escolha pelo menor RMSE",
    "em validação por bloco → cálculo do VIP",
])
arrow([(660,1780),(660,1825)],BLUE)
pred(1825,125,"Selecionar as cinco bandas de maior VIP",[
    "Top 5 específico de cada parâmetro e data",
])
arrow([(660,1950),(660,1995)],BLUE)
pred(1995,155,"PLSR final com as cinco bandas",[
    "Reescolher componentes pelo RMSE em validação por bloco",
    "Predizer deixando um bloco de fora a cada vez",
])
arrow([(660,2150),(660,2195)],BLUE)
pred(2195,170,"Resultados por parâmetro e data",[
    "R² e RMSE de ajuste e de validação por bloco",
    "Valores observados × preditos · resíduos · bandas VIP",
    "CHL total, clorofilas A/B, gs, CRA, ETR, E, A e Ci",
])
box(120,2420,1080,210,"Interpretação da validação",[
    "A seleção de bandas e de componentes usa a data inteira",
    "antes das dobras. As métricas são diagnósticas",
    "e podem ser otimistas; não representam validação externa.",
],BLUE,"#e6effa",33,28)

box(1400,1010,1080,155,"IRRIG × NIRRIG · um modelo por dia",[
    "NIRRIG é a classe positiva · manter as leituras individuais",
    "GroupKFold: cada um dos quatro blocos fica fora uma vez",
],PURPLE,"#f3edf8",35,29)
arrow([(1940,1165),(1940,1200),(1660,1200),(1660,1250)],PURPLE)
arrow([(1940,1200),(2220,1200),(2220,1250)],PURPLE)

def cls(x,y,h,title,lines):
    box(x,y,520,h,title,lines,PURPLE,"#f3edf8",30,27)


cls(1400,1250,205,"Seleção Top 5",[
    "Somente no treino da dobra:",
    "agrupar bandas por Spearman",
    "|ρ| > 0,80 · extensão < 10 nm",
    "ligação completa",
])
arrow([(1660,1455),(1660,1500)],PURPLE)
cls(1400,1500,180,"Bandas representantes",[
    "Banda central de cada grupo",
    "Random Forest → Gini",
    "Selecionar as cinco melhores",
])
arrow([(1660,1680),(1660,1725)],PURPLE)
cls(1400,1725,185,"XGBoost com Top 5",[
    "Treinar na dobra de treino",
    "Prever o bloco excluído",
    "Refazer seleção a cada dobra",
])

cls(1960,1250,205,"Alternativa FWHM",[
    "Centros das feições espectrais",
    "obtidos na média global",
    "417, 730, 1364 e 1956 nm",
    "Quatro bandas fixas",
])
arrow([(2220,1455),(2220,1500)],PURPLE)
cls(1960,1500,180,"Entrada direta no modelo",[
    "Usar as quatro bandas",
    "Sem nova seleção por",
    "Spearman ou Random Forest",
])
arrow([(2220,1680),(2220,1725)],PURPLE)
cls(1960,1725,185,"XGBoost com FWHM",[
    "Treinar na dobra de treino",
    "Prever o bloco excluído",
    "Repetir nos quatro blocos",
])
arrow([(1660,1910),(1660,1950),(1940,1950),(1940,1995)],PURPLE)
arrow([(2220,1910),(2220,1950),(1940,1950)],PURPLE)
box(1400,1995,1080,170,"Resultados de classificação por dia",[
    "Previsões dos blocos excluídos → matriz de confusão",
    "Acurácia · acurácia balanceada · precisão · sensibilidade",
    "F1 · AUC-ROC · estatísticas por classe",
],PURPLE,"#f3edf8",35,29)
arrow([(1940,2165),(1940,2210)],PURPLE)
box(1400,2210,1080,155,"Ajuste final do caminho Top 5",[
    "Refazer Spearman + Random Forest com todas as leituras do dia",
    "Top 5 descritivo → XGBoost final daquele dia",
],PURPLE,"#f3edf8",34,29)
box(1400,2420,1080,210,"Interpretação da validação",[
    "Top 5: seleção refeita somente no treino de cada dobra.",
    "FWHM: centros escolhidos na base inteira; avaliação",
    "exploratória, condicionada às quatro bandas fixas.",
],PURPLE,"#eee5f5",33,28)

text(1300,2730,"Fonte: Pipeline_Final_Predicao.ipynb e Pipeline_Final_Classificacao.ipynb",28,MUTED)
output = ROOT / "fluxograma_predicao_classificacao.png"
im.save(output, dpi=(300,300))
print(output)
print(f"{W} × {H} px · PNG · 300 dpi")
