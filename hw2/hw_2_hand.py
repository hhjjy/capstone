'''
Author: Leo lion24161582@gmail.com
Date: 2024-03-29 20:48:37
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-03-30 01:34:23
FilePath: \capstone\hw2\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import unittest
import numpy as np
import cv2

def filter(img, kernel):
    """
    對圖像應用濾波器。

    :param img: 輸入的灰度圖像，類型為 numpy ndarray。
    :param kernel: 應用於圖像的卷積核，類型為 numpy ndarray。
    :return: 新的經過濾波處理的圖像，類型為 numpy ndarray。
    """
    kernel_size = kernel.shape[0]
    r = kernel_size // 2
    height, width = img.shape[:2]
    padding_img = np.pad(img, ((r, r), (r, r)), 'reflect') # reflect 可以減少邊界誤差
    new_img = np.zeros_like(img)
    for y in range(height):
        for x in range(width):
            region = padding_img[y:y+kernel_size, x:x+kernel_size]
            conv = np.sum(np.multiply(region, kernel))
            new_img[y, x] = conv
    return new_img
def get_guass_kernel(kernel_size, sigma):
    """
    生成高斯核。

    :param kernel_size: 高斯核的大小，類型為整數(3、5、7 等奇數)。
    :param sigma: 高斯核的標準差。
    :return: 生成的高斯核，類型為 numpy ndarray。
    """
    r = int(kernel_size//2)
    kernel = np.zeros((kernel_size, kernel_size))
    for y in range(0-r,0+r+1):
        for x in range(0-r,0+r+1):
            kernel[y+1,x+1] = 1/(2*np.pi*(sigma**2))*np.exp(-(x**2+y**2)/(2*(sigma**2)))
    # 正規化 讓總和為1 
    sum_of_kernel = np.sum(kernel)
    kernel = kernel / sum_of_kernel
    return kernel
def non_maximum_suppression(G, angle):
    """
    非極大值抑制，用於邊緣細化。

    :param G: 梯度幅值圖像，類型為 numpy ndarray。
    :param angle: 梯度方向圖像，以度為單位，類型為 numpy ndarray。
    :return: 經過非極大值抑制處理的圖像，類型為 numpy ndarray。
    """
    M, N = G.shape
    Z = np.zeros_like(G)
    for y in range(1,M-1): 
        for x in range(1,N-1):
            try:
                q = 255
                r = 255
                # 取得theta 對應方向的鄰居q,r數值大小
                if (0<=angle[y,x]<22.5) or (157.5<=angle[y,x]<=180):
                    q = G[y,x-1]
                    r = G[y,x+1]
                elif (22.5<=angle[y,x]<67.5): 
                    q = G[y-1,x+1]
                    r = G[y+1,x-1]
                elif (67.5<=angle[y,x]<112.5): 
                    q = G[y-1,x]
                    r = G[y+1,x]
                elif (112.5<=angle[y,x]<157.5): 
                    q = G[y-1,x-1]
                    r = G[y+1,x+1]

                # 比較 如果當前點比鄰居大 這個點設為有數值(代表是邊緣)
                if (G[y,x]>=q) and (G[y,x]>=r):
                    Z[y,x] = G[y,x]
                else:# 否則數值為0
                    Z[y,x] = 0

            except IndexError as e:
                pass
    return Z
def apply_double_threshold(G, low_threshold, high_threshold):
    """
    應用雙閾值算法進行邊緣檢測。
    規則:
        1. 強邊緣設置為255
        2. 弱邊緣若在強邊緣旁視為強邊緣 設置為255
        3. 低於弱邊緣的數值設為0

    :param G: 梯度幅值圖像，類型為 numpy ndarray。
    :param low_threshold: 低閾值，用於識別弱邊緣。
    :param high_threshold: 高閾值，用於識別強邊緣。
    :return: 經過雙閾值處理的圖像，類型為 numpy ndarray。
    """
    M,N=G.shape[0:2]
    strong_edge = np.zeros_like(G)
    weak_edge = np.zeros_like(G)
    strong_edge[G>=high_threshold] = 1 
    weak_edge[(G<high_threshold) & (G>=low_threshold)] = 1 #界於之間的訊號設為弱訊號
    for y in range(1,M):
        for x in range(1,N):
            if weak_edge[y, x] == 1:
                region = strong_edge[y-1:y+2, x-1:x+2]
                if np.any(region > 0):  # 弱邊緣附近有強邊緣
                    strong_edge[y, x] = 1  # 升級成強邊緣
    return strong_edge*255
if __name__ == "__main__":
    # import os
    # print(os.getcwd())
    # print(os.path.exists('.\\input\\brick.jpg'))

    # 加載圖像並轉成灰階
    img = cv2.imread('.\\input\\lotus.jpg', cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32)#float32避免運算時溢出
    # 高斯濾除雜訊
    gussian = get_guass_kernel(kernel_size=3,sigma=0.3)
    filtered_img = filter(img_float, gussian)
    # Sobel Operator 
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)#float32避免運算時溢出
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)#float32避免運算時溢出
    Gx = filter(filtered_img, sobel_x)
    Gy = filter(filtered_img, sobel_y)
    # Gradient Magnitude and Theta 
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi #逕度轉度數
    # 非極大值抑制=>邊緣細化
    non_maximum_suppression_img = non_maximum_suppression(G,theta)
    # 雙閥值檢測=>保留強邊緣與強邊緣周遭訊號 
    threshold_img = apply_double_threshold(G, 100, 200)
    cv2.imshow('Original Image', img)
    # G_display = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)# # 幅度轉換方便顯示
    # cv2.imshow('Sobel X', cv2.convertScaleAbs(Gx))
    # cv2.imshow('Sobel Y', cv2.convertScaleAbs(Gy))
    # cv2.imshow('Gradient Magnitude', G_display)
    # cv2.imshow('non_maximum_suppression', non_maximum_suppression_img)
    cv2.imshow('threshold_img', threshold_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
