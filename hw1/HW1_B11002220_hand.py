import cv2
import numpy as np
from matplotlib import pyplot as plt
# 計算hist 
def calchist(gray_image):
    hist = np.zeros(256) #空的陣列 0-255
    for h in range(gray_image.shape[0]):
        for w in range(gray_image.shape[1]):
            hist[gray_image[h,w]] += 1 #統計亮度數直
    return hist

# 載入圖片
image = cv2.imread("hw1.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image)
# 顯示原圖 
plt.figure()
plt.title("Before equalizeHist by hand ")
plt.imshow(gray_image, cmap='gray')
plt.show()
# 把二維數據轉成一維直方圖
hist = np.zeros(256) 
for h in range(gray_image.shape[0]):
    for w in range(gray_image.shape[1]):
        hist[gray_image[h,w]] += 1
# 顯示直方圖
plt.figure() #創建一個圖
plt.title("hist by hand ")
plt.xlabel("Bins") 
plt.ylabel("# of Pixels")
plt.plot(hist) #畫一維陣列的圖
plt.xlim([0, 256])#限制x範圍
plt.show()
# 計算並顯示概率質量函數（PMF）
pmf = hist / sum(hist)
# plt.figure()
# plt.title("Probability Mass Function")
# plt.xlabel("Bins")
# plt.ylabel("Probability")
# plt.plot(pmf)
# plt.xlim([0, 256])
# plt.show()

# 計算累積分佈函數
def cumsum(pmf):
    cdf = np.zeros_like(pmf)  # 創建一個與PMF相同大小的零數組，用於存儲CDF的值
    cumulative_sum = 0  
    for i in range(len(pmf)): 
        cumulative_sum += pmf[i] #累加
        cdf[i] = cumulative_sum
    return cdf
cdf = cumsum(pmf) 
# plt.figure()
# plt.title("Cumulative Distribution Function")
# plt.xlabel("Bins")
# plt.ylabel("Probability")
# plt.plot(cdf)
# plt.xlim([0, 256])
# plt.show()

# 換出轉移曲線 cdf[gray_image[height, width]] * (255-0)
# 找到換算的亮度 = 距離差  乘上 放大的比例
# 1. cdf - cdfmin 是指當下機率與最小的機率距離 
# 2.  (255/ (cdf_max - cdf_min) )) 是指這個距離放大的倍數
# 3. pmf 是指該個亮度在整體的佔比
# 4. cdf 是指小於等於該個亮度在整體的佔比 
# 正規化 CDF
cdf_min = cdf[0]
cdf_max = cdf[-1]
cdf_normalized = (cdf - cdf_min) *  255/ (cdf_max - cdf_min)
new_ = np.zeros_like(gray_image) # 創建一個新的二維陣列儲存圖片轉換結果
height, width = gray_image.shape

for w in range(width):
    for h in range(height):
        # 使用正規化後的 CDF 進行映射
        new_[h, w] = cdf_normalized[gray_image[h, w]]

#顯示圖片結果 
plt.figure()
plt.title("After equalizeHist by hand")
plt.imshow(new_, cmap='gray',vmin=0, vmax=255)
plt.show()

# 計算並顯示處理後直方圖
hist = calchist(new_)
plt.figure()
plt.title("hist by hand")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
