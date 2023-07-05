import cv2
import numpy as np

imagem_filtro = cv2.imread('arara_filtro.png', 0) 
imagem_original = cv2.imread('arara.png')
imagem_original_gray = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
filtro_redimensionado = cv2.resize(imagem_filtro, (imagem_original_gray.shape[1], imagem_original_gray.shape[0]))
filtro_fft = np.fft.fft2(filtro_redimensionado)
imagem_fft = np.fft.fft2(imagem_original_gray)
imagem_filtrada_fft = imagem_fft * filtro_fft
imagem_filtrada = np.fft.ifft2(imagem_filtrada_fft)
imagem_filtrada = np.abs(imagem_filtrada)
imagem_filtrada = (imagem_filtrada - np.min(imagem_filtrada)) / (np.max(imagem_filtrada) - np.min(imagem_filtrada))
imagem_filtrada = (imagem_filtrada * 255).astype(np.uint8)
cv2.imwrite('arara_filtrada.png', imagem_filtrada)