import cv2
import numpy as np


def limiarizacao_otsu(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, limiar = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, imagem_limiarizada = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imagem_limiarizada

imagens = ["harewood.jpg", "nuts.jpg", "snow.jpg", "img_aluno.jpg"]

for imagem_path in imagens:
    imagem = cv2.imread(imagem_path)
    imagem_limiarizada = limiarizacao_otsu(imagem)
    nome_arquivo_saida = f"{imagem_path.split('.')[0]}_limiarizada.jpg"
    cv2.imwrite(nome_arquivo_saida, imagem_limiarizada)
