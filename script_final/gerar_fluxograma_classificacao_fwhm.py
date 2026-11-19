"""Exporta o fluxo das seções 4 e 5 do notebook final de classificação."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "fluxograma_classificacao_fwhm.png"
im = Image.new("RGB", (2000, 2800), "white")
d = ImageDraw.Draw(im)
INK, MUTED = "#203a50", "#476173"
GREEN, PURPLE, BLUE = "#167361", "#795092", "#245f9c"


def font(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf"), size)


def text(x, y, s, size=31, color=INK, bold=False):
    d.text((x, y), s, font=font(size, bold), fill=color, anchor="mm")


def box(y, h, title, lines, color=GREEN, fill="#eef8f4"):
    d.rounded_rectangle((200,y,1800,y+h),radius=22,fill=fill,outline=color,width=3)
    items = [(title,36,True)] + [(line,29,False) for line in lines]
    total = sum(size+6 for _,size,_ in items)
    cy = y+(h-total)/2
    for line,size,bold in items:
        text(1000,cy+(size+6)/2,line,size,color if bold else MUTED,bold)
        cy += size+6


def arrow(start,end,color=GREEN):
    d.line((1000,start,1000,end-3),fill=color,width=4)
    d.polygon([(1000,end),(989,end-18),(1011,end-18)],fill=color)


text(1000,67,"Classificação por regiões espectrais (FWHM)",55,INK,True)
text(1000,125,"Regiões → centros das feições → quatro bandas fixas → XGBoost",32,MUTED)

box(190,110,"1. Dados espectrais e metadados",[
    "Unificada13052026.csv · bloco, genótipo, condição e data",
])
arrow(300,335)
box(335,110,"2. Limpeza dos metadados",[
    "Conferir genótipo e condição pela identificação; padronizar blocos",
])
arrow(445,480)
box(480,180,"3. Pré-processamento espectral",[
    "400–2450 nm · remover valores extremos (|valor| > 10.000)",
    "Corrigir emendas em 1000 e 1800 nm · interpolar ausentes",
    "Savitzky–Golay (janela 11, grau 2) → SNV por espectro",
])
arrow(660,695)
box(695,110,"4. Selecionar somente as coletas da manhã",[
    "Manter os metadados e as leituras espectrais individuais",
])
arrow(805,840,PURPLE)
box(840,110,"5. Calcular o espectro médio global",[
    "Reunir todos os dias, genótipos e condições hídricas",
],PURPLE,"#f3edf8")
arrow(950,985,PURPLE)
box(985,140,"6. Delimitar as quatro regiões de análise",[
    "R1: 405–435 nm     ·     R2: 710–750 nm",
    "R3: 1350–1373 nm     ·     R4: 1926–1997 nm",
],PURPLE,"#f3edf8")
arrow(1125,1160,PURPLE)
box(1160,110,"7. Construir a linha de base em cada região",[
    "Reta entre os extremos do intervalo → desvio = curva média − linha de base",
],PURPLE,"#f3edf8")
arrow(1270,1305,PURPLE)
box(1305,140,"8. Identificar o centro da feição espectral",[
    "Comprimento de onda com maior desvio absoluto em relação à linha de base",
    "Desvio positivo: pico · desvio negativo: vale",
],PURPLE,"#f3edf8")
arrow(1445,1480,PURPLE)
box(1480,155,"9. Caracterizar a largura à meia altura (FWHM)",[
    "Localizar, por interpolação, os cruzamentos em 50% da amplitude da feição",
    "FWHM = comprimento de onda à direita − comprimento de onda à esquerda",
    "O centro da feição é a banda usada como entrada na classificação",
],PURPLE,"#f3edf8")
arrow(1635,1670,BLUE)
box(1670,155,"10. Extrair as quatro bandas fixas e separar por dia",[
    "417 nm · 730 nm · 1364 nm · 1956 nm",
    "Três genótipos reunidos por dia · IRRIG = 0 · NIRRIG = 1 (classe positiva)",
    "Entradas do modelo: respostas SNV nas quatro bandas centrais",
],BLUE,"#edf4fc")
arrow(1825,1860,BLUE)
box(1860,195,"11. Validar o XGBoost por bloco em cada dia",[
    "GroupKFold com 4 dobras: treinar em três blocos e prever o bloco excluído",
    "Repetir até cada bloco ter sido previsto fora do treino",
    "Usar as mesmas quatro bandas em todas as dobras",
    "Manter hiperparâmetros e sementes do XGBoost da classificação principal",
],BLUE,"#edf4fc")
arrow(2055,2090,BLUE)
box(2090,155,"12. Avaliar as previsões dos blocos excluídos",[
    "Matriz de confusão · acurácia · acurácia balanceada · precisão",
    "Sensibilidade · F1 · AUC-ROC · estatísticas por classe",
    "Salvar métricas por dia e por dobra, classes e probabilidades previstas",
],BLUE,"#edf4fc")
arrow(2245,2280,BLUE)
box(2280,110,"13. Ajustar o modelo final de cada dia",[
    "Treinar o XGBoost nas quatro bandas com todas as leituras daquele dia",
],BLUE,"#edf4fc")

d.rounded_rectangle((200,2450,1800,2700),22,fill="#f8f3e9",outline="#ad8345",width=2)
text(1000,2490,"Como interpretar este fluxo",35,"#88642e",True)
for y,line in zip([2535,2577,2619,2661],[
    "A FWHM descreve a largura da feição; o modelo utiliza os centros, e não as larguras.",
    "Os centros são definidos com a média global, incluindo os blocos depois usados no teste.",
    "Assim, a avaliação é exploratória e condicionada às quatro bandas já escolhidas.",
    "A classificação com essas bandas não executa seleção por Spearman ou Random Forest.",
]):
    text(1000,y,line,29,MUTED)
text(1000,2750,"Fonte: Pipeline_Final_Classificacao.ipynb · seções 4.0, 5 e 5.1",28,MUTED)
im.save(OUT,dpi=(300,300))
print(OUT)
