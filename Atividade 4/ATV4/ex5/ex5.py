import cv2
import numpy as np


def crescimento_regiao(imagem, semente, limiar):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    mascara = np.zeros_like(imagem_cinza)
    altura, largura = imagem_cinza.shape
    valor_semente = imagem_cinza[semente[1], semente[0]]
    pixels_a_visitar = [semente]

    while len(pixels_a_visitar) > 0:
        x, y = pixels_a_visitar.pop()
        if mascara[y, x] == 0:
            valor_pixel = imagem_cinza[y, x]
            if abs(int(valor_pixel) - int(valor_semente)) <= limiar:
                mascara[y, x] = 255
                if x > 0:
                    pixels_a_visitar.append((x - 1, y))
                if x < largura - 1:
                    pixels_a_visitar.append((x + 1, y))
                if y > 0:
                    pixels_a_visitar.append((x, y - 1))
                if y < altura - 1:
                    pixels_a_visitar.append((x, y + 1))
    imagem_destacada = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2BGR)
    imagem_destacada[mascara != 0] = [0, 0, 255]

    return imagem_destacada


imagem = cv2.imread("root.jpg")
semente = (450, 390)
limiar = 10
imagem_resultante = crescimento_regiao(imagem, semente, limiar)
cv2.imwrite("root_regiao_destacada.jpg", imagem_resultante)
