import numpy as np 
import cv2
import math
import matplotlib.pyplot as plt 
#import cython_ITF
import torch
from statistics import mean, variance

def fourier(data,N,figureplot=False):
    t = np.linspace(0,N,N)

    #plot traject
    if figureplot:
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(8,4))
        fig, (ax1, ax2, ax3) = plt.subplots(3)

        ax1.plot(t,data[:,0])
        ax2.plot(t,data[:,1])
        ax3.plot(t,data[:,2])
        ax1.set_title("X traject",fontsize=10)
        ax2.set_title("Y traject",fontsize=10)
        ax3.set_title("A traject",fontsize=10)
        ax1.set_ylabel('Amplitude')
        ax1.set_xlabel('frames')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('frames')
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('frames')

        fig.savefig('traject_output.png')

    #perform FFT
    fft_X = np.fft.fft(data[:,0])
    fft_Y = np.fft.fft(data[:,1])
    fft_A = np.fft.fft(data[:,2])

    x1 = fft_X[2:7]
    x2 = fft_Y[2:7]
    x3 = fft_A[2:7]
    AC_X = sum(abs(x1)**2)
    total_X = sum(abs(fft_X[2:N//2])**2)
    AC_Y = sum(abs(x2)**2)
    total_Y = sum(abs(fft_Y[2:N//2])**2)
    AC_A = sum(abs(x3)**2)
    total_A = sum(abs(fft_A[2:N//2])**2)

    score_X = AC_X / total_X
    score_Y = AC_Y / total_Y
    score_A = AC_A / total_A

    stable_score = min(score_X,score_Y,score_A)

    return stable_score


def stability_score(input_path):
    vid = cv2.VideoCapture(input_path)
    Nframe = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    cameraPath   = np.empty((Nframe-1,4))
    imgSize=(640,480)

    _,prevFrame = vid.read()
    prevFrame = cv2.resize(prevFrame,imgSize)
    prevGray = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY)
    for idx in range(Nframe-1):
        checkFrame,curFrame = vid.read()
        if not checkFrame:
            break
        curFrame = cv2.resize(curFrame,imgSize)
        curGray = cv2.cvtColor(curFrame,cv2.COLOR_BGR2GRAY)

        prevFtp = cv2.goodFeaturesToTrack(prevGray, maxCorners=300, qualityLevel=0.01,
                                                     minDistance=30, blockSize=3)
        curFtp, status, err = cv2.calcOpticalFlowPyrLK(prevGray,curGray,prevFtp,None)
        i = np.where(status==1)[0]
        prevFtp = prevFtp[i]
        curFtp = curFtp[i]

        H,_ = cv2.estimateAffinePartial2D(prevFtp,curFtp)
        dx = H[0,2]
        dy = H[1,2]
        da = np.arctan2(H[1][0],H[0][0])
        ds = np.sqrt(H[1][0]**2 + H[0][0]**2)

        if idx==0:
            cameraPath[idx,:] = [dx, dy, da, ds]
        else:
            x = cameraPath[idx-1, 0] + dx
            y = cameraPath[idx-1, 1] + dy
            a = cameraPath[idx-1, 2] + da
            s = cameraPath[idx-1, 3] * ds
            cameraPath[idx,:] = [x, y, a, s]
        prevFrame = curFrame.copy()
        prevGray = curGray.copy()

    stable_score = fourier(cameraPath,Nframe-1)
    return stable_score



def PSNR(img1,img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    w,h = img1.shape
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    diff = img1-img2

    error = np.sum(diff**2)/(w*h)
    psnr = (10*math.log10((255**2)/error))

    return psnr

def PSNR_WMR(img1,img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
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



def ITF_score(input_path):
    vid = cv2.VideoCapture(input_path)
    Nframe = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ITF = 0
    _,prevFrame = vid.read()
    prevFrame = cv2.resize(prevFrame,(640,360))
    for idx in range(Nframe-1):
        checkFrame,curFrame = vid.read()
        curFrame = cv2.resize(curFrame,(640,360))
        if not checkFrame:
            break
        psnr = PSNR(prevFrame,curFrame)
        ITF += psnr
        prevFrame = curFrame.copy()
    ITF = ITF/(Nframe-1)
    return ITF

# def ITF_WMR_score(input_path):
#     vid = cv2.VideoCapture(input_path)
#     Nframe = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#     ITF = 0
#     _,prevFrame = vid.read()
#     prevFrame = cv2.resize(prevFrame,(640,360))
#     for idx in range(Nframe-1):
#         checkFrame,curFrame = vid.read()
#         curFrame = cv2.resize(curFrame,(640,360))
#         if not checkFrame:
#             break
#         img1 = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY)
#         img2 = cv2.cvtColor(curFrame,cv2.COLOR_BGR2GRAY)
#         psnr = cython_ITF.PSNR_cython(img1,img2)
#         ITF += psnr
#         prevFrame = curFrame.copy()
#     ITF = ITF/(Nframe-1)
#     return ITF


def distortion_cropping(src_path,dst_path):
    vid_src = cv2.VideoCapture(src_path)
    vid_dst = cv2.VideoCapture(dst_path)
    Nframe = int(vid_dst.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_src_ratio = 0
    crop_dst_ratio = 0

    distortion_ratio = 10000
    imgSize=(640,360)
    Totalsize = 640*360
    print('Nframe ',Nframe)
    list_crop = []
    list_distor = []
    count = 0
    for idx in range(Nframe):
        checkSrcFrame,srcFrame = vid_src.read()
        checkDstFrame,dstFrame = vid_dst.read()
        if not checkDstFrame or not checkSrcFrame:
            break
        count +=1
        srcFrame = cv2.resize(srcFrame,imgSize)
        dstFrame = cv2.resize(dstFrame,imgSize)
        srcGray = cv2.cvtColor(srcFrame,cv2.COLOR_BGR2GRAY)
        dstGray = cv2.cvtColor(dstFrame,cv2.COLOR_BGR2GRAY)

        src_pts = cv2.goodFeaturesToTrack(srcGray, maxCorners=200, qualityLevel=0.01,
                                                     minDistance=30, blockSize=3)
        dst_pts, status, err = cv2.calcOpticalFlowPyrLK(srcGray,dstGray,src_pts,None)
        i = np.where(status==1)[0]
        src_pts = src_pts[i]
        dst_pts = dst_pts[i]

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        A = H[0:2,0:2]
        U, s, V = np.linalg.svd(A)
        distortion = s[1]/s[0]
        distortion_ratio = min(distortion_ratio,distortion)

        # src_blk_pixel = np.sum(srcGray==0)
        # dst_blk_pixel = np.sum(dstGray==0)
        src_blk_pixel = cv2.countNonZero(srcGray)
        dst_blk_pixel = cv2.countNonZero(dstGray)

        crop_src =  src_blk_pixel/Totalsize
        crop_dst =  dst_blk_pixel/Totalsize
        crop_src_ratio += crop_src
        crop_dst_ratio += crop_dst
        list_crop.append(crop_dst)
        list_distor.append(distortion)


    #cropping_score = (crop_dst_ratio/(count)) / (crop_src_ratio/(count))
    cropping_score = (crop_dst_ratio/(count))
    var_crop = variance(list_crop)
    var_distor = variance(list_distor)
    mean_crop = mean(list_crop)
    mean_distor = mean(list_distor)
    min_crop = min(list_crop)
    min_distor = min(list_distor)
    max_crop = max(list_crop)
    max_distor = max(list_distor)


    return [min_crop,max_crop,mean_crop,var_crop],[min_distor,max_distor,mean_distor,var_distor]


def crop_video(src_path,dst_path,name):
    vid_src = cv2.VideoCapture(src_path)
    vid_dst = cv2.VideoCapture(dst_path)
    Nframe = int(vid_dst.get(cv2.CAP_PROP_FRAME_COUNT))
    imgSize=(640,360)
    saveVid = cv2.VideoWriter('crop_'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30,imgSize)
    w = 640
    h= 360
    crop_ratio = 0.7
    w_crop = int(crop_ratio*w)
    h_crop = int(crop_ratio*h)
    startX = int((w - w_crop)/2)
    endX   = int(startX + w_crop)
    startY = int((h - h_crop)/2)
    endY   = int(startY + h_crop)

    xmin = startX
    xmax = endX
    ymin = startY
    ymax = endY
    for idx in range(Nframe):
        checkSrcFrame,srcFrame = vid_src.read()
        checkDstFrame,dstFrame = vid_dst.read()
        if not checkDstFrame:
            break
        srcFrame = cv2.resize(srcFrame,imgSize)
        dstFrame = cv2.resize(dstFrame,imgSize)
        srcGray = cv2.cvtColor(srcFrame,cv2.COLOR_BGR2GRAY)
        dstGray = cv2.cvtColor(dstFrame,cv2.COLOR_BGR2GRAY)

        src_pts = cv2.goodFeaturesToTrack(srcGray, maxCorners=200, qualityLevel=0.01,
                                                     minDistance=30, blockSize=3)
        dst_pts, status, err = cv2.calcOpticalFlowPyrLK(srcGray,dstGray,src_pts,None)
        i = np.where(status==1)[0]
        src_pts = src_pts[i]
        dst_pts = dst_pts[i]

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        corners = np.array([
                              [0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]
                           ])

        corners = cv2.perspectiveTransform(np.float32([corners]), H) 


        dx = H[0,2]
        dy = H[1,2]
        if dx>0:
            x_left = max(corners[0][0,0],corners[0][1,0])
            xmin = max(x_left,xmin)
            xmax = xmin + w_crop
            if xmax>w-1:
                xmax = w-1
                xmin = max(xmax -w_crop,x_left)

        elif dx<0:
            x_right = min(corners[0][2,0],corners[0][3,0])
            xmax = min(x_right,xmax)
            xmin = xmax - w_crop
            if xmin<0:
                xmin = 0
                xmax = min(xmin + w_crop,x_right)

        if dy>0:
            y_up = max(corners[0][0,1],corners[0][3,1])
            ymin = max(ymin,y_up)
            ymax = ymin + h_crop
            if ymax>h-1:
                ymax = h-1
                ymin = max(ymax-h_crop,y_up)

        elif dy<0:
            y_down = min(corners[0][1,1],corners[0][2,1])
            ymax = min(y_down,ymax)
            ymin = ymax - h_crop
            if ymin<0:
                ymin=0
                ymax = min(ymin + h_crop,y_down)

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        cropImg = dstFrame[ymin:ymax, xmin:xmax]
        cropImg = cv2.resize(cropImg,imgSize)

        saveVid.write(cropImg)


path = '../out_regular_walker_origin.avi'
score = ITF_score(path)
print(score)