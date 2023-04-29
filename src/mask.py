import numpy as np
import cv2

def get_mask(ambient,flash):
    DIFF = flash-ambient
    mean_D = np.mean(DIFF)
    std_D = np.std(DIFF)
    k = 1.5  
    S = (DIFF < mean_D - k * std_D)
    P = (DIFF > mean_D + k * std_D)
    M = S.astype(np.float32) + P.astype(np.float32)
    M = cv2.threshold(M, 0.5, 1.0, cv2.THRESH_BINARY)[1]

    return M

# Load the input image
ambient  = cv2.imread('data\\lamp\\lamp_ambient.tif').astype(np.float32)
flash    = cv2.imread('data\\lamp\\lamp_flash.tif').astype(np.float32)
M=get_mask(ambient,flash)
A_DETAIL = cv2.imread('data\\lamp\\Bilater_flitering\\detail_bilateral_space15_intensity05.jpg').astype(np.float32)
A_BASE   = cv2.imread('data\\lamp\\Bilater_flitering\\bilateral_space15_intensity25.jpg').astype(np.float32)
ONE    = np.ones_like(M)
AFinal = np.multiply(np.subtract(ONE,M),A_DETAIL)+np.multiply(M,A_BASE)
AFinal = AFinal.astype(np.uint8)
cv2.imwrite("data\\lamp\\Bilater_flitering\\masking.jpg",AFinal)