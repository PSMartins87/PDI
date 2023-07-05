import cv2
import numpy as np

image = cv2.imread('teste.png', 0)  # Carregar em escala de cinza

# Filtro passa-baixa 
kernel_size_pb = 3
kernel_pb = np.ones((kernel_size_pb, kernel_size_pb), np.float32) / (kernel_size_pb * kernel_size_pb)
image_pb = cv2.filter2D(image, -1, kernel_pb)

# Filtro passa-alta 
kernel_size_pa = 3
kernel_pa = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
image_pa = cv2.filter2D(image, -1, kernel_pa)

# Filtro passa-banda
kernel_size_pb1 = 5
kernel_pb1 = np.ones((kernel_size_pb1, kernel_size_pb1), np.float32) / (kernel_size_pb1 * kernel_size_pb1)
image_pb1 = cv2.filter2D(image, -1, kernel_pb1)

kernel_size_pa1 = 3
kernel_pa1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
image_pa1 = cv2.filter2D(image_pb1, -1, kernel_pa1)

image_pb_band = image_pb1 - image_pa1

# Filtro rejeita-banda
kernel_size_pa2 = 5
kernel_pa2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
image_pa2 = cv2.filter2D(image, -1, kernel_pa2)

kernel_size_pb2 = 3
kernel_pb2 = np.ones((kernel_size_pb2, kernel_size_pb2), np.float32) / (kernel_size_pb2 * kernel_size_pb2)
image_pb2 = cv2.filter2D(image_pa2, -1, kernel_pb2)

image_rb = image_pa2 - image_pb2

cv2.imwrite('passa_baixa.png', image_pb)
cv2.imwrite('passa_alta.png', image_pa)
cv2.imwrite('passa_banda.png', image_pb_band)
cv2.imwrite('rejeita_banda.png', image_rb)
