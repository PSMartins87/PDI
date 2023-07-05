import cv2
import numpy as np

imagem = cv2.imread('circuito.tif', 0)

for i in range(3):
    imagem_filtrada = cv2.medianBlur(imagem, 3)  
    cv2.imwrite(f'circuito_mediana{i+1}.tif', imagem_filtrada) 
    imagem = imagem_filtrada
