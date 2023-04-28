import numpy as np
import cv2

def enforce_boundary_conditions(I_init, I, B):
    H,W = I_init.shape[:2]
    for C in range(I_init.shape[2]):
        I_init[:B,:,C] = I[:B,:,C]
        I_init[H-B+1:H+1,:,C] = I[H-B+1:H+1,:,C]
        I_init[:,:B,C] = I[:,:B,C]
        I_init[:,W-B+1:W+1,C] = I[:,W-B+1:W+1,C] 
    return I_init

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

def laplacian_filtering(I_star):
    """
    Compute the Laplacian of I_star using a 5x5 filter.
    """
    laplacian_kernel = np.array([[0, 0, -1, 0, 0],
                                 [0, -1, -2, -1, 0],
                                 [-1, -2, 16, -2, -1],
                                 [0, -1, -2, -1, 0],
                                 [0, 0, -1, 0, 0]])
    return cv2.filter2D(I_star, -1, laplacian_kernel)

def CGD(div, I_star_init,I ,B, eps, N):
    """
    Perform gradient field integration using conjugate gradient descent.
    """
    # Initialization
    I_star = enforce_boundary_conditions(I_star_init,I, B)
    r = div - laplacian_filtering(I_star)
    d = r
    delta_new = np.sum(r * r)
    n = 0

    # Conjugate gradient descent iteration
    while delta_new > eps**2 and n < N:
        q = laplacian_filtering(d)
        eta = delta_new / np.sum(d * q)
        I_star = enforce_boundary_conditions(I_star + eta * d, B)
        r = r - eta * q
        delta_old = delta_new
        delta_new = np.sum(r * r)
        beta = delta_new / delta_old
        d = r + beta * d
        n += 1

    return I_star

if __name__=="__main__":
    ambient = cv2.imread('data\\museum\\museum_ambient.png')
    flash   = cv2.imread("data\\museum\\museum_flash.png")
    gray_ambient = cv2.cvtColor(ambient, cv2.COLOR_BGR2GRAY)
    gray_flash = cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)

    Ix_ambient = cv2.Sobel(gray_ambient, cv2.CV_64F, 1, 0, ksize=3)
    Iy_ambient = cv2.Sobel(gray_ambient, cv2.CV_64F, 0, 1, ksize=3)
    Ixy_ambient = cv2.merge((Ix_ambient, Iy_ambient))

    Ix_flash = cv2.Sobel(gray_flash, cv2.CV_64F, 1, 0, ksize=3)
    Iy_flash = cv2.Sobel(gray_flash, cv2.CV_64F, 0, 1, ksize=3)
    Ixy_flash = cv2.merge((Ix_flash, Iy_flash))

    div_ambient = cv2.Laplacian(Ix_ambient, cv2.CV_64F) + cv2.Laplacian(Iy_ambient, cv2.CV_64F)

    I_init = np.zeros_like(ambient)

    B=4
    sigma = 0.2
    tau_s = 0.8
    N=3
    I_init_start=enforce_boundary_conditions(I_init,ambient,B)
    img = CGD(div_ambient,I_init,ambient,B,sigma,N)
    cv2.imshow("d",img)
    cv2.waitKey(0)
    M = compute_M(Ixy_ambient,Ixy_flash)

    sigma = 0.2
    tau_s = 0.8
    W = compute_W(gray_flash,sigma,tau_s)
    D = compute_NEW_D(W,Ixy_ambient,Ixy_flash,M)
    print(D.shape)