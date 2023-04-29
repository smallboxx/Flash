import numpy as np
import cv2
from showdetail import showdetails,print_progress_bar

def EnforceBoundaryConditions(I_init, B):
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

def compute_M(Ixy_ambient,Ixy_flash):
    ep=1e-9
    Ix_ambient = Ixy_ambient[:,:,0]
    Iy_ambient = Ixy_ambient[:,:,1]
    Ix_flash   = Ixy_flash[:,:,0]
    Iy_flash   = Ixy_flash[:,:,1]
    molecular  = np.abs(np.multiply(Ix_flash,Ix_ambient)+np.multiply(Iy_flash,Iy_ambient))
    denominator= np.multiply(np.sqrt(np.square(Ix_flash)+np.square(Iy_flash)),np.sqrt(np.square(Ix_ambient)+np.square(Iy_ambient)))
    M = molecular/(denominator+ep)
    return M

def compute_W(flash,sigma,tau_s):
    # Compute the saturation weight map
    ws = np.tanh(sigma * (flash - tau_s))

    # Normalize the saturation map to [0, 1]
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws))
    return ws

def compute_NEW_D(ws,Ixy_ambient,Ixy_flash,M):
    Ix_ambient = Ixy_ambient[:,:,0]
    Iy_ambient = Ixy_ambient[:,:,1]
    Ix_flash   = Ixy_flash[:,:,0]
    Iy_flash   = Ixy_flash[:,:,1]
    E = np.ones_like(ws)
    Ix=np.multiply(ws,Ix_ambient)+np.multiply((E-ws),np.multiply(M,Ix_flash)+np.multiply((E-M),Ix_ambient))
    Iy=np.multiply(ws,Iy_ambient)+np.multiply((E-ws),np.multiply(M,Iy_flash)+np.multiply((E-M),Iy_ambient))
    D = np.dstack([Ix,Iy])
    return D

def LaplacianFiltering(I_star):
    laplacian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filtered_I = cv2.filter2D(I_star, -1, laplacian_kernel)
    return filtered_I

def CGD(div_I, I_star_init, B, eps=1e-100, N=5000):
    I_star = EnforceBoundaryConditions(I_star_init, B)
    r = div_I - LaplacianFiltering(I_star)
    d = r
    delta_new = np.sum(np.multiply(r,r))
    n = 0
    while delta_new > eps**2 and n < N:
        print(delta_new)
        q = LaplacianFiltering(d)
        eta = delta_new / np.sum(np.multiply(d,q))
        I_star = EnforceBoundaryConditions(I_star + eta*d, B)
        r = r - eta*q
        delta_old = delta_new
        delta_new = np.sum(np.multiply(r,r))
        beta = delta_new / delta_old
        d = r + beta*d
        n += 1

    return I_star

if __name__=="__main__":
    ambient = cv2.imread('data\\museum\\museum_ambient.png')
    flash   = cv2.imread("data\\museum\\museum_flash.png")
    gray_ambient = cv2.cvtColor(ambient,cv2.COLOR_BGR2GRAY)
    div=compute_div(gray_ambient)
    I_init = np.zeros_like(gray_ambient)
    div_restore = CGD(div,I_init,gray_ambient)

    cv2.imshow("org",gray_ambient)
    cv2.imshow("img",div_restore)
    cv2.waitKey(0)
    # M = compute_M(Ixy_ambient,Ixy_flash)

    # sigma = 0.2
    # tau_s = 0.8
    # W = compute_W(gray_flash,sigma,tau_s)
    # D = compute_NEW_D(W,Ixy_ambient,Ixy_flash,M)
    # print(D.shape)