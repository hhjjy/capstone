import unittest
import numpy as np
import cv2

def filter(img, kernel):
    kernel_size = kernel.shape[0]
    r = kernel_size // 2
    height, width = img.shape[:2]
    padding_img = np.pad(img, ((r, r), (r, r)), 'reflect')
    new_img = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            region = padding_img[y:y+kernel_size, x:x+kernel_size]
            conv = np.sum(np.multiply(region, kernel))
            new_img[y, x] = conv
    return new_img

class SobelFilterTest(unittest.TestCase):
    def test_sobel_filter(self):
        sobel_GX = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
        matches = 0
        for _ in range(10000):
            # 随机生成图像大小
            rows, cols = np.random.randint(3, 10, size=2)
            A = np.random.randint(0, 256, (rows, cols)).astype(np.float32)
            
            test_sobel_cv2 = cv2.filter2D(A, -1, sobel_GX)
            test_sobel_mine = filter(A, sobel_GX)
            
            if np.allclose(test_sobel_cv2, test_sobel_mine, atol=1e-6):
                matches += 1
        
        print(f"Number of matches: {matches} out of 10000")
        self.assertTrue(matches > 0, "No matches found.")

if __name__ == "__main__":
    unittest.main()
