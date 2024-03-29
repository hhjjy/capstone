# import cv2
# import numpy as np
# from matplotlib import pyplot as plt


# class ImageProcessor:
#     def __init__(self):
#         self.img = None
#         # self.gary_img = gary_img

#     @staticmethod
#     def show_img(img, title, xlabel, ylabel):
#         plt.figure()
#         plt.imshow(img, cmap='gray')
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.show()

#     @staticmethod
#     def add_salt_and_pepper(img,fraction_salt:float,fraction_pepper:float):
#         '''
#         img:輸入的灰階圖像
#         fraction_salt:整體影像的鹽比例255
#         fraction_pepper:整體影像的胡椒比例
#         return: 新的圖像
#         '''
#         row , col = img.shape[0:2]
#         num_salt = int(np.ceil(fraction_salt * img.size ))#無條件進位並轉成整數
#         num_pepper = int(np.ceil(fraction_pepper * img.size ))
#         rand_arr = np.random.rand(img.shape[0],img.shape[1])
#         # 複製原本圖片 
#         output = img.copy()
#         # randarr < fraction_salt 產生一個二維陣列 只要符合這個條件的都是true 
#         # randarr > (1-fraction_pepper) 產生一個二維陣列 只要符合這個條件的都是true 
#         output[rand_arr < fraction_salt] = 255
#         output[rand_arr > (1-fraction_pepper)] = 0
#         return output    
    


#     def median_filter(self, kernel_size):
#         pass

#     def mean_filter(self, kernel_size):
#         pass
    
#     @staticmethod
#     def get_guass_kernel(kernel_size, sigma):
#         r = int(kernel_size//2)
#         kernel = np.zeros((kernel_size, kernel_size))
#         for y in range(0-r,0+r+1):
#             for x in range(0-r,0+r+1):
#                 kernel[y+1,x+1] = 1/(2*np.pi*(sigma**2))*np.exp(-(x**2+y**2)/(2*(sigma**2)))
#         # normalize 
#         sum_of_kernel = np.sum(kernel)
#         kernel = kernel / sum_of_kernel
#         return kernel
#     @staticmethod
#     def gussian_filter(img,kernel_size,sigma):
#         """高斯濾波"""
#         kernel = self.get_guass_kernel(kernel_size, sigma)
#         r = kernel_size //2 # 圖片半徑 r=2 
#         height,width = img.shape[0:2]
#         # 填充 
#         padding_img = np.pad(img,(r,r)) 
#         new_img = np.zeros_like(img)
#         for y in range(r,height+r):
#             for x in range(r,width+r):
#                 region = padding_img[(y-r):(y+r+1),(x-r):(x+r+1)]# y ,x = 2-2,2+2+1
#                 conv = np.sum(region*kernel)
#                 new_img[y-2,x-2] = conv
#         return new_img


#     def bilinear_interpolate(img,scale_factor):
#         src_height, src_width = img.shape
#         dst_height, dst_width = src_height * scale_factor, src_width * scale_factor
#         dst_img = np.zeros((dst_height, dst_width), dtype=np.uint8)
#         for y in range(dst_height):
#             for x in range(dst_width):
#                 # 1. 把新的圖的座標轉回原始座標
#                 src_x = x/scale_factor # 3.3
#                 src_y = y/scale_factor
#                 # 2. 找出鄰近四個點 
#                 x1,y1 = np.floor([src_x, src_y]).astype(int)
#                 x2,y2 = np.ceil([src_x, src_y]).astype(int)
#                 # x2 = min(x2, src_width-1)# 確保不超過邊界 
#                 # y2 = min(y2, src_height-1)# 確保不超過邊界 
#                 dx = src_x - x1 # 0.3 距離Q1 x = 0.3 
#                 dy = src_y - y1 # 0.1 距離Q1 y = 0.1
#                 Q11 = img[y1,x1]
#                 Q12 = img[y1,x2]
#                 Q21 = img[y2,x1]
#                 Q22 = img[y2,x2]
#                 value = (1-dx)*(1-dy)*Q11 + dx*(1-dy)*Q12 + (1-dx)*dy*Q21 + dx*dy*Q22
#                 dst_img[y, x] = int(value)
#         return dst_img
