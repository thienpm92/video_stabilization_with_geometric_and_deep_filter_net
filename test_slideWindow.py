import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv

from model.filter_model import FilterNet
from model.cnn_geometric_model import GeometricMatching_model
from model.loss import TransformedGridLoss
from data.synth_dataset import SynthDataset
from geotnf.transformation import SynthPairTnf
from data.normalization import NormalizeImageDict

def main():
    #srcPath = '/media/vipl/DATA/Dataset/video/dataset/regular_outdoor_origin.avi'
    srcPath = '/media/vipl/DATA/Dataset/video_stab/DeepStab-dataset/unstable/25.avi'
    savePath = 'out_regular_outdoor.avi'
    geometric_path = './model/GeometricMatching_model.pth'
    deepfilter_path = './model/filter_model.pth'
    imgSize = (640,320)
    winSize = 150
    overlap = 30
    stride = winSize - overlap
    saveVid = cv2.VideoWriter(savePath,cv2.VideoWriter_fourcc('M','J','P','G'),30,imgSize)
    vid = cv2.VideoCapture(srcPath)
    Nframe = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


    orgTransform = np.empty((Nframe-1,2,3))
    cameraPath   = np.empty((Nframe-1,2,3))
    smoothPath   = np.empty((Nframe-1,6))

    state = torch.load(geometric_path)
    geometric_model = GeometricMatching_model(use_cuda=True,
                                                output_dim=6,
                                                feature_extraction_cnn='resnet101',
                                                matching_type='correlation')             
    geometric_model.load_state_dict(state['state_dict'])
    geometric_model.eval()


    deepfilter_model = FilterNet()
    state = torch.load(deepfilter_path)
    deepfilter_model.load_state_dict(state['state_dict'])
    deepfilter_model.cuda()
    deepfilter_model.eval()

    print('Generate trajectories')
    _,prevFrame = vid.read()
    prevFrame = cv2.resize(prevFrame,imgSize)
    prevGray = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY)
    with torch.no_grad():
        for idx in range(Nframe-1):
            checkFrame,curFrame = vid.read()
            if not checkFrame:
                break
            curFrame = cv2.resize(curFrame,imgSize)
            curGray = cv2.cvtColor(curFrame,cv2.COLOR_BGR2GRAY)

            prevFtp = cv2.goodFeaturesToTrack(prevGray, maxCorners=200, qualityLevel=0.01,
                                                        minDistance=30, blockSize=3)
            curFtp, status, err = cv2.calcOpticalFlowPyrLK(prevGray,curGray,prevFtp,None)
            i = np.where(status==1)[0]
            prevFtp = prevFtp[i]
            curFtp = curFtp[i]

            H,_ = cv2.estimateAffine2D(prevFtp,curFtp)
            A = H[0:2,0:2]
            t = H[0:2,2:3]
            orgTransform[idx,:] = H
            if idx==0:
                cameraPath[idx,:] = H
            else:
                a = np.matmul(cameraPath[idx-1, 0:2, 0:2], A)
                t = cameraPath[idx-1, 0:2, 2:3] + t
                cameraPath[idx,:] = np.hstack((a,t))
            prevFrame = curFrame.copy()
            prevGray = curGray.copy()
        vid.release()

    print('smooth path')
    trajectA,trajectB,trajectX,trajectC,trajectD,trajectY = cameraPath.reshape(-1,6).T

    Nstep = round((Nframe-1) / stride)
    for idx in range(Nstep):
        startFrame = idx * stride
        endFrame = startFrame + winSize
        if idx == Nstep-1:  
            smoothPath[startFrame:startFrame+overlap,0] = buffer_A[:,0]
            smoothPath[startFrame:startFrame+overlap,1] = buffer_B[:,0]
            smoothPath[startFrame:startFrame+overlap,2] = buffer_X[:,0]
            smoothPath[startFrame:startFrame+overlap,3] = buffer_C[:,0]
            smoothPath[startFrame:startFrame+overlap,4] = buffer_D[:,0]
            smoothPath[startFrame:startFrame+overlap,5] = buffer_Y[:,0]

            endFrame = Nframe-1 - 1
            startFrame = endFrame - winSize
            l = ((idx-1)*stride + winSize)-startFrame-1

            buffer_X = out_X[winSize-l:]
            buffer_Y = out_Y[winSize-l:]
            buffer_A = out_A[winSize-l:]
            buffer_B = out_B[winSize-l:]
            buffer_C = out_C[winSize-l:]
            buffer_D = out_D[winSize-l:]


        X = trajectX[startFrame:endFrame]
        Y = trajectY[startFrame:endFrame]
        A = trajectA[startFrame:endFrame]
        B = trajectB[startFrame:endFrame]
        C = trajectC[startFrame:endFrame]
        D = trajectD[startFrame:endFrame]
            
        X = torch.from_numpy(X.reshape(X.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        Y = torch.from_numpy(Y.reshape(Y.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        A = torch.from_numpy(A.reshape(A.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        B = torch.from_numpy(B.reshape(B.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        C = torch.from_numpy(C.reshape(C.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        D = torch.from_numpy(D.reshape(D.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)

        tmp_X = deepfilter_model(X)
        tmp_Y = deepfilter_model(Y)
        tmp_A = deepfilter_model(A)
        tmp_B = deepfilter_model(B)
        tmp_C = deepfilter_model(C)
        tmp_D = deepfilter_model(D)
            
        out_X = tmp_X.cpu().squeeze(0).permute(1,0).data.numpy()
        out_Y = tmp_Y.cpu().squeeze(0).permute(1,0).data.numpy()
        out_A = tmp_A.cpu().squeeze(0).permute(1,0).data.numpy()
        out_B = tmp_B.cpu().squeeze(0).permute(1,0).data.numpy()
        out_C = tmp_C.cpu().squeeze(0).permute(1,0).data.numpy()
        out_D = tmp_D.cpu().squeeze(0).permute(1,0).data.numpy()

        if idx == 0:    #first window
            smoothPath[0:winSize-overlap,0] = out_A[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,1] = out_B[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,2] = out_X[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,3] = out_C[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,4] = out_D[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,5] = out_Y[0:winSize-overlap,0]

            buffer_X = out_X[winSize-overlap:]
            buffer_Y = out_Y[winSize-overlap:]
            buffer_A = out_A[winSize-overlap:]
            buffer_B = out_B[winSize-overlap:]
            buffer_C = out_C[winSize-overlap:]
            buffer_D = out_D[winSize-overlap:]

        elif idx == Nstep-1:    #last window
            w1 = np.array([i/l for i in range(0,l)]).reshape(-1,1)
            w2 = np.array([1 - x for x in w1]).reshape(-1,1)

            tmpX_ = np.multiply(out_X[:l],w1) + np.multiply(buffer_X,w2)
            tmpY_ = np.multiply(out_Y[:l],w1) + np.multiply(buffer_Y,w2)
            tmpA_ = np.multiply(out_A[:l],w1) + np.multiply(buffer_A,w2)
            tmpB_ = np.multiply(out_B[:l],w1) + np.multiply(buffer_B,w2)
            tmpC_ = np.multiply(out_C[:l],w1) + np.multiply(buffer_C,w2)
            tmpD_ = np.multiply(out_D[:l],w1) + np.multiply(buffer_D,w2)

            smoothPath[startFrame:startFrame+l,0] = tmpA_[:,0]
            smoothPath[startFrame:startFrame+l,1] = tmpB_[:,0]
            smoothPath[startFrame:startFrame+l,2] = tmpX_[:,0]
            smoothPath[startFrame:startFrame+l,3] = tmpC_[:,0]
            smoothPath[startFrame:startFrame+l,4] = tmpD_[:,0]
            smoothPath[startFrame:startFrame+l,5] = tmpY_[:,0]

            smoothPath[startFrame+l+1:,0] = out_A[l:,0]
            smoothPath[startFrame+l+1:,1] = out_B[l:,0]
            smoothPath[startFrame+l+1:,2] = out_X[l:,0]
            smoothPath[startFrame+l+1:,3] = out_C[l:,0]
            smoothPath[startFrame+l+1:,4] = out_D[l:,0]
            smoothPath[startFrame+l+1:,5] = out_Y[l:,0]

        else:   #normal window
            w1 = np.array([i/overlap for i in range(0,overlap)]).reshape(-1,1)
            w2 = np.array([1 - x for x in w1]).reshape(-1,1)
            
            tmpX_ = np.multiply(out_X[:overlap],w1) + np.multiply(buffer_X,w2)
            tmpY_ = np.multiply(out_Y[:overlap],w1) + np.multiply(buffer_Y,w2)
            tmpA_ = np.multiply(out_A[:overlap],w1) + np.multiply(buffer_A,w2)
            tmpB_ = np.multiply(out_B[:overlap],w1) + np.multiply(buffer_B,w2)
            tmpC_ = np.multiply(out_C[:overlap],w1) + np.multiply(buffer_C,w2)
            tmpD_ = np.multiply(out_D[:overlap],w1) + np.multiply(buffer_D,w2)

            smoothPath[startFrame:startFrame+overlap,0] = tmpA_[:,0]
            smoothPath[startFrame:startFrame+overlap,1] = tmpB_[:,0]
            smoothPath[startFrame:startFrame+overlap,2] = tmpX_[:,0]
            smoothPath[startFrame:startFrame+overlap,3] = tmpC_[:,0]
            smoothPath[startFrame:startFrame+overlap,4] = tmpD_[:,0]
            smoothPath[startFrame:startFrame+overlap,5] = tmpY_[:,0]

            smoothPath[startFrame+overlap:startFrame+winSize-overlap,0] = out_A[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,1] = out_B[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,2] = out_X[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,3] = out_C[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,4] = out_D[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,5] = out_Y[overlap:winSize-overlap,0]
        
            buffer_X = out_X[winSize-overlap:]
            buffer_Y = out_Y[winSize-overlap:]
            buffer_A = out_A[winSize-overlap:]
            buffer_B = out_B[winSize-overlap:]
            buffer_C = out_C[winSize-overlap:]
            buffer_D = out_D[winSize-overlap:]

    print('warping')
    vid = cv2.VideoCapture(srcPath)
    smoothPath = smoothPath.reshape(-1,2,3)
    for idx in range(Nframe-1):
        checkFrame,img = vid.read()
        if not checkFrame:
            break
        img = cv2.resize(img,imgSize)
        A = np.matmul(orgTransform[idx,0:2,0:2] , np.matmul(smoothPath[idx,0:2,0:2] , inv(cameraPath[idx,0:2,0:2])))
        t = orgTransform[idx, 0:2, 2:3] + smoothPath[idx, 0:2, 2:3] - cameraPath[idx, 0:2, 2:3]
        H = np.hstack((A,t))

        newImg = cv2.warpAffine(img,H,imgSize)
        saveVid.write(newImg)

    saveVid.release()
    vid.release()

    t = np.arange(0,Nframe-1)
    fig,axs = plt.subplots(1)
    fig.suptitle('trajectory')
    plt.plot(t,cameraPath[:,0,1])
    plt.plot(t,smoothPath[:,0,1])
    plt.show()


if __name__ == '__main__':
    main()






