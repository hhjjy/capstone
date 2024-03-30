import cv2
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """將圖像用matplotlib顯示出來"""
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def canny_edge_detection(image_path):
    # 讀取原始圖像
    img = cv2.imread(image_path)
    
    # 轉換為灰度圖像，Canny邊緣檢測需要灰度圖
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 應用Canny邊緣檢測
    # 第一和第二個參數分別是低閾值和高閾值
    # 第三個參數設定梯度大小的孔徑，默認為3
    edges = cv2.Canny(gray_img, 100, 200, apertureSize=3)
    
    plt.figure(figsize=(10, 5))
    cv2.imshow("Original",gray_img)
    cv2.imshow("Canny Edge Detection",edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # show_img_with_matplotlib(img, "Original Image", 1)
    # show_img_with_matplotlib(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Edge Detection Result", 2)
    # plt.show()

image_path = '.\input\lotus.jpg'
canny_edge_detection(image_path)
