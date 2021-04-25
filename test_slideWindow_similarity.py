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
    #srcPath = '/media/vipl/DATA/Dataset/video/dataset/regular_street_origin.avi'
    srcPath = '/media/vipl/DATA/Dataset/video_stab/DeepStab-dataset/unstable/25.avi'
    savePath = 'out_25.avi'
    geometric_path = './model/GeometricMatching_model.pth'
    deepfilter_path = './model/filter_model.pth'
    imgSize = (256,256)
    winSize = 150
    overlap = 30
    stride = winSize - overlap
    saveVid = cv2.VideoWriter(savePath,cv2.VideoWriter_fourcc('M','J','P','G'),30,imgSize)
    vid = cv2.VideoCapture(srcPath)
    Nframe = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


    orgTransform = np.empty((Nframe-1,4))
    cameraPath   = np.empty((Nframe-1,4))
    smoothPath   = np.empty((Nframe-1,4))

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

            H,_ = cv2.estimateAffinePartial2D(prevFtp,curFtp)
            dx = H[0,2]
            dy = H[1,2]
            da = np.arctan2(H[1][0],H[0][0])
            ds = np.sqrt(H[1][0]**2 + H[0][0]**2)
            orgTransform[idx,:] = [dx,dy,da,ds]
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
        vid.release()

    print('smooth path')
    trajectX,trajectY,trajectA,trajectS = cameraPath.reshape(-1,4).T

    Nstep = round((Nframe-1) / stride)
    for idx in range(Nstep):
        startFrame = idx * stride
        endFrame = startFrame + winSize
        if idx == Nstep-1:  
            smoothPath[startFrame:startFrame+overlap,0] = buffer_X[:,0]
            smoothPath[startFrame:startFrame+overlap,1] = buffer_Y[:,0]
            smoothPath[startFrame:startFrame+overlap,2] = buffer_A[:,0]
            smoothPath[startFrame:startFrame+overlap,3] = buffer_S[:,0]

            endFrame = Nframe-1 - 1
            startFrame = endFrame - winSize
            l = ((idx-1)*stride + winSize)-startFrame-1

            buffer_X = out_X[winSize-l:]
            buffer_Y = out_Y[winSize-l:]
            buffer_A = out_A[winSize-l:]
            buffer_S = out_S[winSize-l:]

        X = trajectX[startFrame:endFrame]
        Y = trajectY[startFrame:endFrame]
        A = trajectA[startFrame:endFrame]
        S = trajectS[startFrame:endFrame]
            
        X = torch.from_numpy(X.reshape(X.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        Y = torch.from_numpy(Y.reshape(Y.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        A = torch.from_numpy(A.reshape(A.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)
        S = torch.from_numpy(S.reshape(S.shape[0],1).astype(float)).float().cuda().permute(1,0).unsqueeze(0)

        tmp_X = deepfilter_model(X)
        tmp_Y = deepfilter_model(Y)
        tmp_A = deepfilter_model(A)
        tmp_S = deepfilter_model(S)
            
        out_X = tmp_X.cpu().squeeze(0).permute(1,0).data.numpy()
        out_Y = tmp_Y.cpu().squeeze(0).permute(1,0).data.numpy()
        out_A = tmp_A.cpu().squeeze(0).permute(1,0).data.numpy()
        out_S = tmp_S.cpu().squeeze(0).permute(1,0).data.numpy()

        if idx == 0:    #first window
            smoothPath[0:winSize-overlap,0] = out_X[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,1] = out_Y[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,2] = out_A[0:winSize-overlap,0]
            smoothPath[0:winSize-overlap,3] = out_S[0:winSize-overlap,0]

            buffer_X = out_X[winSize-overlap:]
            buffer_Y = out_Y[winSize-overlap:]
            buffer_A = out_A[winSize-overlap:]
            buffer_S = out_S[winSize-overlap:]

        elif idx == Nstep-1:    #last window
            w1 = np.array([i/l for i in range(0,l)]).reshape(-1,1)
            w2 = np.array([1 - x for x in w1]).reshape(-1,1)

            tmpX_ = np.multiply(out_X[:l],w1) + np.multiply(buffer_X,w2)
            tmpY_ = np.multiply(out_Y[:l],w1) + np.multiply(buffer_Y,w2)
            tmpA_ = np.multiply(out_A[:l],w1) + np.multiply(buffer_A,w2)
            tmpS_ = np.multiply(out_S[:l],w1) + np.multiply(buffer_S,w2)

            smoothPath[startFrame:startFrame+l,0] = tmpX_[:,0]
            smoothPath[startFrame:startFrame+l,1] = tmpY_[:,0]
            smoothPath[startFrame:startFrame+l,2] = tmpA_[:,0]
            smoothPath[startFrame:startFrame+l,3] = tmpS_[:,0]

            smoothPath[startFrame+l+1:,0] = out_X[l:,0]
            smoothPath[startFrame+l+1:,1] = out_Y[l:,0]
            smoothPath[startFrame+l+1:,2] = out_A[l:,0]
            smoothPath[startFrame+l+1:,3] = out_S[l:,0]

        else:   #normal window
            w1 = np.array([i/overlap for i in range(0,overlap)]).reshape(-1,1)
            w2 = np.array([1 - x for x in w1]).reshape(-1,1)
            
            tmpX_ = np.multiply(out_X[:overlap],w1) + np.multiply(buffer_X,w2)
            tmpY_ = np.multiply(out_Y[:overlap],w1) + np.multiply(buffer_Y,w2)
            tmpA_ = np.multiply(out_A[:overlap],w1) + np.multiply(buffer_A,w2)
            tmpS_ = np.multiply(out_S[:overlap],w1) + np.multiply(buffer_S,w2)

            smoothPath[startFrame:startFrame+overlap,0] = tmpX_[:,0]
            smoothPath[startFrame:startFrame+overlap,1] = tmpY_[:,0]
            smoothPath[startFrame:startFrame+overlap,2] = tmpA_[:,0]
            smoothPath[startFrame:startFrame+overlap,3] = tmpS_[:,0]
            
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,0] = out_X[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,1] = out_Y[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,2] = out_A[overlap:winSize-overlap,0]
            smoothPath[startFrame+overlap:startFrame+winSize-overlap,3] = out_S[overlap:winSize-overlap,0]
        
            buffer_X = out_X[winSize-overlap:]
            buffer_Y = out_Y[winSize-overlap:]
            buffer_A = out_A[winSize-overlap:]
            buffer_S = out_S[winSize-overlap:]


    print('warping')
    vid = cv2.VideoCapture(srcPath)
        
    for idx in range(Nframe-1):
        checkFrame,img = vid.read()
        if not checkFrame:
            break
        img = cv2.resize(img,imgSize)
        newX = orgTransform[idx,0] + smoothPath[idx,0] - cameraPath[idx,0]
        newY = orgTransform[idx,1] + smoothPath[idx,1] - cameraPath[idx,1]
        newA = orgTransform[idx,2] + smoothPath[idx,2] - cameraPath[idx,2]
        newS = orgTransform[idx,3] * smoothPath[idx,3] / cameraPath[idx,3]

        H = np.array(([np.cos(newA)*newS, -np.sin(newA)*newS, newX] ,
                          [np.sin(newA)*newS,  np.cos(newA)*newS, newY]))

        newImg = cv2.warpAffine(img,H,imgSize)
        saveVid.write(newImg)

    saveVid.release()
    vid.release()



    t = np.arange(0,Nframe-1)
    fig,axs = plt.subplots(1)
    fig.suptitle('trajectory')
    plt.plot(t,cameraPath[:,1])
    plt.plot(t,smoothPath[:,1])
    plt.show()


if __name__ == '__main__':
    main()






