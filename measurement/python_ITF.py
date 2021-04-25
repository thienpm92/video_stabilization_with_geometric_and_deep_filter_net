import cv2
import numpy as np
import math
def PSNR_python(img1,img2):
    h,w = img1.shape
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    N_pixel = w*h
    tmp = w*h
    total = 0
    for i in range (h):
        for j in range (w):
            if (img1[i,j] ==0) or (img2[i,j] ==0):
                N_pixel -= 1
                continue
            total += abs(img1[i,j]-img2[i,j])**2
    error = total/N_pixel
    psnr = (10*math.log10((255**2)/error))
    return psnr