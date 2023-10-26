
import cv2

import matplotlib.pyplot as plt

import numpy as np

image = cv2.imread(r'G:\Applied_AI_Project\neutral\4.jpg', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

gauss = cv2.GaussianBlur(image, (0, 0), 2.0)
image_unsharp_boosting= cv2.addWeighted(image, 2.0, gauss, -1.0, 0)

normalized_image = image_unsharp_boosting / 255.0


plt.figure(figsize=(12, 6))  
plt.subplot(2, 2, 1) 
plt.imshow(image,cmap='gray')
plt.title('Normal Image')


plt.subplot(2, 2, 2)  
plt.imshow(blurred_image,cmap='gray')
plt.title('blurred Image')


plt.subplot(2, 2, 3)  
plt.imshow(image_unsharp_boosting,cmap='gray')
plt.title('Sharped Image')

plt.subplot(2, 2, 4)  
plt.imshow(normalized_image,cmap='gray')
plt.title('Normalize image ')

plt.show()
