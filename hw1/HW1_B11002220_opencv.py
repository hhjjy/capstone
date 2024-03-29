import cv2
import numpy as np
from matplotlib import pyplot as plt

# 載入圖片
image = cv2.imread("hw1.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 顯示原圖片 & 直方圖 
plt.figure()
plt.title("Before equalizeHist by opencv ")
plt.imshow(gray_image, cmap='gray')
plt.show()
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist.flatten()
plt.figure()
plt.title("hist by opencv ")
plt.xlabel("Bins")
plt.ylabel("numbers of piexls ")
plt.plot(hist)
plt.show()

# 轉換成等化直方圖
eqHist = cv2.equalizeHist(gray_image)
# 顯示轉換後 & 直方圖
plt.figure()
plt.title("After equalizeHist by opencv")
plt.imshow(eqHist, cmap='gray')
plt.show()

hist = cv2.calcHist([eqHist], [0], None, [256], [0, 256])
hist.flatten()# 二維轉一維才能顯示
# print(hist)
plt.figure()
plt.title("hist by opencv")
plt.xlabel("Bins")
plt.ylabel("numbers of piexls ")
plt.plot(hist)
plt.show()