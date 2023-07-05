import cv2
image = cv2.imread('teste.png')

kernel_size = (5, 5)  
sigma = 1.0 

filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
cv2.imwrite('filtro_gaussiano.png', filtered_image)
cv2.imwrite('resultado_gaussiano.png', image)
