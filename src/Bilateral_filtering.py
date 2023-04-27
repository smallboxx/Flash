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


def bilateral_filter(L, diameter, sigma_color, sigma_space):
    output = np.zeros_like(L)
    radius = diameter // 2
    L= np.divide(L,255)
    for i in range(radius,L.shape[0]-radius):
        for j in range(radius,L.shape[1]-radius):
            pixel_value = 0.0
            weight_sum = 0.0
            for k in range(i - radius, i + radius + 1):
                for l in range(j - radius, j + radius + 1):
                    # 计算空间距离
                    spatial_distance = pow((i - k),2) + pow((j - l),2)

                    # 计算颜色距离
                    color_distance =pow((L[i, j] - L[k, l]),2)

                    # 计算权重
                    weight = np.exp(-spatial_distance/ (2 * sigma_space ** 2))
                    weight *= np.exp(-color_distance / (2 * sigma_color ** 2))

                    # 累加像素值和权重
                    pixel_value += weight * L[k, l]
                    weight_sum += weight
            output[i, j] = pixel_value / weight_sum
            print_progress_bar((i-25)*(L.shape[1]-diameter)+(j-25)+1,(L.shape[1]-diameter)*(L.shape[0]-diameter),"start","complete")
    output=np.uint8(np.multiply(output,255))
    return output

if __name__=="__main__":
    diameter = 50
    input = cv2.imread("data\\lamp\\lamp_ambient.tif",cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    input=cv2.resize(input,[200,200])
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