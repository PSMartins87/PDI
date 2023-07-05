import cv2
import numpy as np

imagem = cv2.imread('igreja.png', 0) 
imagem_suavizada = cv2.GaussianBlur(imagem, (5, 5), 0)
gradiente_x = cv2.filter2D(imagem_suavizada, -1, np.array([[-1, 0, 1]]))
gradiente_y = cv2.filter2D(imagem_suavizada, -1, np.array([[-1], [0], [1]]))
magnitude_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
direcao_gradiente = np.arctan2(gradiente_y, gradiente_x)

bordas = cv2.Canny(imagem_suavizada, 100, 200)
T1 = 5
T2 = 10
bordas_fortes = np.zeros_like(bordas)
bordas_fortes[bordas >= T2] = 255
vizinhos = cv2.dilate(bordas_fortes, None)
bordas_fortes[vizinhos >= T1] = 255
cv2.imwrite('igreja_bordas.png', bordas_fortes)
