import numpy as np
import cv2
import matplotlib.pyplot as plt


def salvar_imagem_espectro(imagem, espectro, caminho_salvar):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(imagem, cmap="gray")
    axs[0].axis("off")
    axs[1].imshow(espectro, cmap="gray")
    axs[1].axis("off")
    plt.tight_layout()
    plt.savefig(caminho_salvar)
    plt.close()


caminhos_imagens = [
    "arara.png",
    "barra1.png",
    "barra2.png",
    "barra3.png",
    "barra4.png",
    "quadrado.png",
    "teste.png",
]

for caminho in caminhos_imagens:
    img = cv2.imread(caminho, 0)
    if img is None:
        print(f"Erro ao carregar a imagem: {caminho}")
        continue
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)
    magnitude_spectrum = 20 * np.log(np.abs(F_shift) + 1e-8)
    nome_arquivo = caminho.split(".")[0] + "_espectro.png"
    salvar_imagem_espectro(img, magnitude_spectrum, nome_arquivo)
