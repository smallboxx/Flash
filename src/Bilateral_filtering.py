import cv2
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
import math
# progress_bar
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

# 定义一个辅助函数，用于计算像素之间的空间距离
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 定义一个辅助函数，用于计算像素之间的灰度值相似度
def similarity(pixel1, pixel2, sigmaColor):
    diff = np.subtract(pixel1, pixel2, dtype=np.float64)
    return np.exp(-(np.sum(diff**2)) / (2 * sigmaColor**2))

def bilateral_filter(L, diameter, sigma_color, sigma_space):
        # 获取输入图像的高度和宽度
    height, width = L.shape[:2]

    # 创建一个空白的输出图像
    img_filtered = np.zeros_like(L)

    # 对于每个像素，计算权重并进行加权平均
    for i in range(height):
        for j in range(width):
            # 计算当前像素的权重
            pixel_weight = np.float64(0)
            pixel_sum = np.zeros(3, dtype=np.float64)
            for k in range(height):
                for l in range(width):
                    # 计算空间距离和灰度值相似度
                    d = distance(i, j, k, l)
                    if d <= diameter:
                        s = similarity(L[i, j], L[k, l], sigma_color)
                        # 计算像素的权重
                        w = s * np.exp(-(d**2) / (2 * sigma_space**2))
                        # 计算加权平均
                        pixel_sum += w * L[k, l]
                        pixel_weight += w
            # 将加权平均结果保存到输出图像中
            img_filtered[i, j] = int(pixel_sum / pixel_weight)
            print([i,j],[height,width])
    return img_filtered

if __name__=="__main__":
    diameter = 50
    input = cv2.imread("data\\lamp\\lamp_ambient.tif",cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    input = cv2.cvtColor(input,cv2.COLOR_LBGR2Lab)
    L=input[:,:,0]
    a=input[:,:,1]
    b=input[:,:,2]
    SPACE_K_LIST=[5,10,20,30,40,50,60]
    INTENSITY_K_LIST=[0.05,0.10,0.15,0.20,0.25]
    for sigma_space in SPACE_K_LIST:
        for sigma_color in INTENSITY_K_LIST:
            Bilateral_L = bilateral_filter(L, diameter, sigma_color, sigma_space)
            #Bilateral_L = cv2.bilateralFilter(L, diameter, sigma_color, sigma_space)
            Bilateral_IMG=cv2.cvtColor(cv2.merge([Bilateral_L,a,b]), cv2.COLOR_Lab2BGR)
            savename="data\\lamp\\Bilater_flitering\\"+"bilateral_space"+str(sigma_space)+"_intensity"+str(sigma_color).split(".")[1]+".tif"
            cv2.imwrite(savename, Bilateral_IMG, [cv2.IMWRITE_TIFF_COMPRESSION, 1])