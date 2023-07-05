import cv2
import numpy as np

imagem = cv2.imread("linhas.png", 0)

template_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
template_horizontal = np.array(
    [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32
)
template_45neg = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)
template_45pos = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)

resultado_vertical = cv2.filter2D(imagem, -1, template_vertical)
resultado_horizontal = cv2.filter2D(imagem, -1, template_horizontal)
resultado_45neg = cv2.filter2D(imagem, -1, template_45neg)
resultado_45pos = cv2.filter2D(imagem, -1, template_45pos)

limiar = 100
limiarizado_vertical = cv2.threshold(resultado_vertical, limiar, 255, cv2.THRESH_BINARY)[1]
limiarizado_horizontal = cv2.threshold(resultado_horizontal, limiar, 255, cv2.THRESH_BINARY)[1]
limiarizado_45neg = cv2.threshold(resultado_45neg, limiar, 255, cv2.THRESH_BINARY)[1]
limiarizado_45pos = cv2.threshold(resultado_45pos, limiar, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite("linhas_vertical.png", limiarizado_vertical)
cv2.imwrite("linhas_horizontal.png", limiarizado_horizontal)
cv2.imwrite("linhas_45neg.png", limiarizado_45neg)
cv2.imwrite("linhas_45pos.png", limiarizado_45pos)
