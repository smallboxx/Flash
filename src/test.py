import cv2
import numpy as np
from scipy.sparse import block_diag
from scipy.sparse.linalg import cg


def enforce_boundary_conditions(I_init, B):
    I_init[0, :] = B[0, :]
    I_init[-1, :] = B[-1, :]
    I_init[:, 0] = B[:, 0]
    I_init[:, -1] = B[:, -1]
    return I_init


def compute_div(I):
    # 计算梯度场
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    grad_I = np.dstack((Ix, Iy))

    # 计算散度
    Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
    div = cv2.add(Ixx, Iyy)
    return div


def restore_image(div, block_size=32):
    height, width = div.shape

    # 创建分块矩阵 A
    A_blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = np.zeros((block_size, block_size))
            if i > 0:
                block += np.eye(block_size)
            if j > 0:
                block += np.eye(block_size, k=1)
            if i < height - block_size:
                block -= np.eye(block_size, k=block_size)
            if j < width - block_size:
                block -= np.eye(block_size, k=block_size-1)
            A_blocks.append(block)
    A = block_diag(A_blocks)

    # 初始化还原的图像
    restored_image = np.zeros((height, width))

    # 使用共轭梯度法求解线性系统 Ax = b
    b = div.ravel()
    x, info = cg(A, b)

    # 将解向量 x 转换为灰度图像
    restored_image = x.reshape((height, width))

    # 将还原的图像限制在像素值的有效范围内
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)

    return restored_image


if __name__ == "__main__":
    # 读入灰度图像
    gray_image = cv2.imread("data\\museum\\museum_ambient.png", cv2.IMREAD_GRAYSCALE)

    # 计算灰度图像的散度场
    div = compute_div(gray_image)

    # 使用共轭梯度法恢复灰度图像
    restored_image = restore_image(div)

    # 显示原始图像和恢复图像
    cv2.imshow("Original", gray_image)
    cv2.imshow("Restored", restored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()