import cv2
import numpy as np

# Define as cores que queremos detectar e seus intervalos em RGB
cores = {
    "verde": ([0, 100, 0], [130, 255, 50]),
    "vermelho": ([0, 0, 100], [95, 85, 255]),
    "azul": ([100, 0, 0], [255, 80, 80]),
    "marrom": ([0, 50, 50], [50, 100, 100]),
    "amarelo": ([0, 200, 200], [50, 255, 255]),
    "laranja": ([0, 100, 200], [50, 150, 255]),
    "roxo": ([100, 0, 100], [255, 50, 255])
}

# Carrega a imagem e converte para o espaco de cor RGB
imagem = cv2.imread(r'/home/me15degrees/Programação/colorblindness/content/0.png')
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Inicializa um dicionario para armazenar a contagem de pixels de cada cor
contagem = {cor: 0 for cor in cores}

# Percorre cada pixel da imagem e verifica em qual intervalo de cor ele se encaixa
altura, largura, _ = imagem.shape
total_pixels = altura * largura
for y in range(altura):
    for x in range(largura):
        pixel = imagem[y, x]
        for cor, (minimo, maximo) in cores.items():
            if np.all(pixel >= minimo) and np.all(pixel <= maximo):
                contagem[cor] += 1
                break

# Calcula a porcentagem de cada cor na imagem
porcentagens = {}
for cor, quantidade in contagem.items():
    porcentagens[cor] = (quantidade / total_pixels) * 100

# Imprime os resultados
for cor, porcentagem in porcentagens.items():
    print(f"{cor}: {porcentagem:.2f}%")
