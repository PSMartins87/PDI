import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

pasta_destino = '//home/paulo/Área de Trabalho/PDI_Exercicios'

img1 = cv2.cvtColor(cv2.imread("w.jpeg"), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread("w.jpg"), cv2.COLOR_BGR2GRAY)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))


def detector_bordas_sobel(imagem):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    height, width = imagem.shape
    gradient_x = np.zeros((height, width))
    gradient_y = np.zeros((height, width))

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gradient_x[y, x] = np.sum(imagem[y - 1 : y + 2, x - 1 : x + 2] * sobel_x)
            gradient_y[y, x] = np.sum(imagem[y - 1 : y + 2, x - 1 : x + 2] * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    return gradient_magnitude


def vizinhanca(image):
    smoothed = np.zeros_like(image)
    height, width = image.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            avg = np.mean(image[y - 1 : y + 2, x - 1 : x + 2])
            smoothed[y, x] = avg
    return smoothed


def filtro_suavizacao_media(imagem, k):
    imagem_suavizada = imagem.copy()
    altura, largura = imagem.shape[:2]
    for y in range(altura):
        for x in range(largura):
            vizinhanca = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    if (x + i >= 0 and x + i < largura) and (
                        y + j >= 0 and y + j < altura
                    ):
                        vizinhanca.append(imagem[y + j, x + i])

            media_vizinhos = np.mean(vizinhanca, axis=0)
            imagem_suavizada[y, x] = media_vizinhos

    return imagem_suavizada


def operador_laplaciano(imagem):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    padded_image = np.pad(imagem, pad_width=1, mode="constant")
    laplacian_img = np.zeros_like(imagem, dtype=np.float32)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            neighborhood = padded_image[i - 1 : i + 2, j - 1 : j + 2]
            laplacian_img[i - 1, j - 1] = np.sum(neighborhood * laplacian_kernel)
    laplacian_img = np.abs(laplacian_img)
    laplacian_img = cv2.normalize(
        laplacian_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    return laplacian_img


def detector_bordas_roberts(imagem):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    height, width = imagem.shape
    gradient_x = np.zeros((height, width))
    gradient_y = np.zeros((height, width))

    for y in range(height - 1):
        for x in range(width - 1):
            gradient_x[y, x] = np.sum(imagem[y : y + 2, x : x + 2] * roberts_x)
            gradient_y[y, x] = np.sum(imagem[y : y + 2, x : x + 2] * roberts_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    return gradient_magnitude


def detector_bordas_prewitt(imagem):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    height, width = imagem.shape
    gradient_x = np.zeros((height, width))
    gradient_y = np.zeros((height, width))

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gradient_x[y, x] = np.sum(imagem[y - 1 : y + 2, x - 1 : x + 2] * prewitt_x)
            gradient_y[y, x] = np.sum(imagem[y - 1 : y + 2, x - 1 : x + 2] * prewitt_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    return gradient_magnitude


img1_suave = vizinhanca(img1)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_suave, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img1_suave.jpeg")
cv2.imwrite(caminho_imagem, img1_suave)

img2_suave = vizinhanca(img2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2_suave, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img2_suave.jpg")
cv2.imwrite(caminho_imagem, img2_suave)


k = 5
img2_k = filtro_suavizacao_media(img2, k)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2_k, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img2_k.jpg")
cv2.imwrite(caminho_imagem, img2_k)

img1_k = filtro_suavizacao_media(img1, k)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_k, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img1_k.jpeg")
cv2.imwrite(caminho_imagem, img1_k)


img1_lap = operador_laplaciano(img1)
img2_lap = operador_laplaciano(img2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_lap, cv2.COLOR_BGR2RGB))

caminho_imagem = os.path.join(pasta_destino, "img1_lap.jpeg")
cv2.imwrite(caminho_imagem, img1_lap)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2_lap, cv2.COLOR_BGR2RGB))

caminho_imagem = os.path.join(pasta_destino, "img2_lap.jpg")
cv2.imwrite(caminho_imagem, img2_lap)

img1_rob = detector_bordas_roberts(img1)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_rob, cv2.COLOR_BGR2RGB))

caminho_imagem = os.path.join(pasta_destino, "img1_rob.jpeg")
cv2.imwrite(caminho_imagem, img1_rob)
img2_rob = detector_bordas_roberts(img2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2_rob, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img2_rob.jpg")
cv2.imwrite(caminho_imagem, img2_rob)


img1_pre = detector_bordas_prewitt(img1)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_pre, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img1_pre.jpeg")
cv2.imwrite(caminho_imagem, img1_pre)

img2_pre = detector_bordas_prewitt(img2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2_pre, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img2_pre.jpg")
cv2.imwrite(caminho_imagem, img2_pre)


img1_sob = detector_bordas_sobel(img1)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_sob, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img1_sob.jpeg")
cv2.imwrite(caminho_imagem, img1_sob)

img2_sob = detector_bordas_sobel(img2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2_sob, cv2.COLOR_BGR2RGB))
caminho_imagem = os.path.join(pasta_destino, "img2_sob.jpg")
cv2.imwrite(caminho_imagem, img2_sob)

