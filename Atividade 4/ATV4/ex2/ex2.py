import cv2
import numpy as np

imagem = cv2.imread("pontos.png", 0)  
filtro = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
imagem_filtrada = cv2.filter2D(imagem, -1, filtro)
limiar = 100  
imagem_binarizada = cv2.threshold(imagem_filtrada, limiar, 255, cv2.THRESH_BINARY)[1]
contornos, _ = cv2.findContours(
    imagem_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.imwrite("pontos_binarizada.png", imagem_binarizada)

