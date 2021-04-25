import cv2
import matplotlib.pyplot as plt 
import cython
import numpy as np
import math

@cython.boundscheck(False)
# cpdef unsigned char[:, :] threshold_fast(int T, unsigned char [:, :] image):
#     # set the variable extension types
#     cdef int x, y, w, h
    
#     # grab the image dimensions
#     h = image.shape[0]
#     w = image.shape[1]
    
#     # loop over the image
#     for y in range(0, h):
#         for x in range(0, w):
#             # threshold the pixel
#             image[y, x] = 255 if image[y, x] >= T else 0
    
#     # return the thresholded image
#     return image



cpdef float PSNR_cython(unsigned char [:, :] img1, unsigned char [:, :] img2):

    cdef int i,j,h,w
    h = img1.shape[0]
    w = img1.shape[1]
    # img1 = img1.astype(np.float32)
    # img2 = img2.astype(np.float32)
    cdef int N_pixel = w*h
    cdef int tmp = w*h
    cdef float total = 0

    for i in range (h):
        for j in range (w):
            if (img1[i,j] ==0) or (img2[i,j] ==0):
                N_pixel -= 1
                continue
            total += float(abs(img1[i,j]-img2[i,j]))**2
    cdef float error = total/N_pixel
    cdef float psnr = (10*math.log10((255**2)/error))
    return psnr
