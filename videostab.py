import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from numpy.linalg import inv

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable





class Videostab:
    def __init__(self,src_path, save_path, traject_method='similarity',smooth='mean',type_matrix='D1',winSize=30,imgSize=(320,240), model_path=None,use_cuda=True):
        self.winSize = winSize
        self.traject_method = traject_method
        self.smooth = smooth
        self.type_matrix = type_matrix
        self.imgSize = imgSize
        self.src_path = src_path
        self.saveVid = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'),30,self.imgSize)
        self.vid = cv2.VideoCapture(src_path)
        self.Nframe = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.use_cuda = use_cuda
        if self.use_cuda:
            system_mode = "gpu"
        else:
            system_mode = "cpu"

        if traject_method[:9] !='geometric' and model_path is not None:
            raise Exception('Setup wrong mode')

        if self.traject_method == 'affine':
            self.orgTransform = np.empty((self.Nframe-1,2,3))
            self.cameraPath   = np.empty((self.Nframe-1,2,3))
            self.smoothPath   = np.empty((self.Nframe-1,2,3))

        elif self.traject_method == 'similarity':
            self.orgTransform = np.empty((self.Nframe-1,4))
            self.cameraPath   = np.empty((self.Nframe-1,4))
            self.smoothPath   = np.empty((self.Nframe-1,4))

        elif self.traject_method == 'homography':
            self.orgTransform = np.empty((self.Nframe-1,3,3))
            self.cameraPath   = np.empty((self.Nframe-1,3,3))
            self.smoothPath   = np.empty((self.Nframe-1,3,3))

        elif self.traject_method == 'geometric_1':      #deep homography_algorithm - affine transform
            self.orgTransform = np.empty((self.Nframe-1,2,3))
            self.cameraPath   = np.empty((self.Nframe-1,2,3))
            self.smoothPath   = np.empty((self.Nframe-1,2,3))
            self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])
            state = torch.load(model_path,map_location=system_mode)
            self.model = DeepHomography_model()
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()

            w,h  = self.imgSize
            patchSize   = 128
            self.left   = w/2 - patchSize/2
            self.up     = h/2 - patchSize/2
            self.right  = self.left + patchSize
            self.bottom = self.up + patchSize
            topLeft_point    = (self.up,self.left)
            botLeft_point   = (self.bottom, self.left)
            botRight_point = (self.bottom, self.right)
            topRight_point  = (self.up, self.right)
            self.four_points = [topLeft_point, botLeft_point, botRight_point, topRight_point]

        elif self.traject_method == 'geometric_2':      #geometric_matching_algorithm - affine transform
            self.orgTransform = np.empty((self.Nframe-1,2,3))
            self.cameraPath   = np.empty((self.Nframe-1,2,3))
            self.smoothPath   = np.empty((self.Nframe-1,2,3))
            self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])

            state = torch.load(model_path,map_location=system_mode)

            feature_extraction_cnn = 'resnet101'
            matching_type='correlation'
            self.geometric_model = GeometricMatching_model(use_cuda=self.use_cuda,
                                                output_dim=6,
                                                feature_extraction_cnn=feature_extraction_cnn,
                                                matching_type=matching_type)             
            self.geometric_model.load_state_dict(state['state_dict'])
            self.geometric_model.eval()


    def trajectory(self):
        _,prevFrame = self.vid.read()
        prevFrame = cv2.resize(prevFrame,self.imgSize)
        prevGray = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            for idx in range(self.Nframe-1):
                checkFrame,curFrame = self.vid.read()
                if not checkFrame:
                    break
                curFrame = cv2.resize(curFrame,self.imgSize)
                curGray = cv2.cvtColor(curFrame,cv2.COLOR_BGR2GRAY)

                prevFtp = cv2.goodFeaturesToTrack(prevGray, maxCorners=200, qualityLevel=0.01,
                                                            minDistance=30, blockSize=3)
                curFtp, status, err = cv2.calcOpticalFlowPyrLK(prevGray,curGray,prevFtp,None)
                i = np.where(status==1)[0]
                prevFtp = prevFtp[i]
                curFtp = curFtp[i]

                if self.traject_method=='affine':
                    H,_ = cv2.estimateAffine2D(prevFtp,curFtp)
                    A = H[0:2,0:2]
                    t = H[0:2,2:3]
                    self.orgTransform[idx,:] = H
                    if idx==0:
                        self.cameraPath[idx,:] = H
                    else:
                        a = np.matmul(self.cameraPath[idx-1, 0:2, 0:2], A)
                        t = self.cameraPath[idx-1, 0:2, 2:3] + t
                        self.cameraPath[idx,:] = np.hstack((a,t))

                elif self.traject_method=='similarity':
                    H,_ = cv2.estimateAffinePartial2D(prevFtp,curFtp)
                    dx = H[0,2]
                    dy = H[1,2]
                    da = np.arctan2(H[1][0],H[0][0])
                    ds = np.sqrt(H[1][0]**2 + H[0][0]**2)
                    self.orgTransform[idx,:] = [dx,dy,da,ds]
                    if idx==0:
                        self.cameraPath[idx,:] = [dx, dy, da, ds]
                    else:
                        x = self.cameraPath[idx-1, 0] + dx
                        y = self.cameraPath[idx-1, 1] + dy
                        a = self.cameraPath[idx-1, 2] + da
                        s = self.cameraPath[idx-1, 3] * ds
                        self.cameraPath[idx,:] = [x, y, a, s]

                elif self.traject_method=='geometric_1':
                    Ip1 = prevGray.copy()
                    Ip2 = curGray.copy()
                    Img1 = Ip1[int(self.up):int(self.bottom),int(self.left):int(self.right)]
                    Img2 = Ip2[int(self.up):int(self.bottom),int(self.left):int(self.right)]
                    training_image = np.dstack((Img1, Img2))
                    training_image = torch.from_numpy(training_image/255.).unsqueeze(0).float()
                    training_image = training_image.permute(0,3,1,2)

                    outputs = self.model(training_image)
                    #outputs = outputs*32
                    #delta = np.int32(outputs.reshape((4,2)))
                    delta = outputs.reshape((4,2))*32
                    print(delta)
                    perturbed_four_ = np.add(np.array(self.four_points),delta)

                    perturbed_four = perturbed_four_.tolist()
                    H,inlier = cv2.estimateAffinePartial2D(np.float32(self.four_points), np.float32(perturbed_four) , False)

                    A = H[0:2,0:2]
                    t = H[0:2,2:3]
                    self.orgTransform[idx,:] = H
                    if idx==0:
                        self.cameraPath[idx,:] = H
                    else:
                        a = np.matmul(self.cameraPath[idx-1, 0:2, 0:2], A)
                        t = self.cameraPath[idx-1, 0:2, 2:3] + t
                        self.cameraPath[idx,:] = np.hstack((a,t))

                elif self.traject_method=='geometric_2':
                    Ip1 = cv2.resize(prevFrame,(240,240))
                    Ip2 = cv2.resize(curFrame,(240,240))

                    Ip1 = torch.from_numpy(Ip1/255.).permute(2,0,1)
                    Ip2 = torch.from_numpy(Ip2/255.).permute(2,0,1)
                    Ip1 = Ip1.unsqueeze(0).float()
                    Ip2 = Ip2.unsqueeze(0).float()
                    
                    batch = {'source_image':Ip1,'target_image':Ip2}
                    
                    theta = self.geometric_model(batch)

                    H = theta.reshape(2,3).numpy()
                    A = H[0:2,0:2]
                    t = H[0:2,2:3]
                    self.orgTransform[idx,:] = np.hstack((A,t))
                    if idx==0:
                        self.cameraPath[idx,:] = np.hstack((A,t))
                    else:
                        a = np.matmul(self.cameraPath[idx-1, 0:2, 0:2], A)
                        t = self.cameraPath[idx-1, 0:2, 2:3] + t
                        self.cameraPath[idx,:] = np.hstack((a,t))

                prevFrame = curFrame.copy()
                prevGray = curGray.copy()
        self.vid.release()
        return self.orgTransform, self.cameraPath






    def smoothTraject(self,method='mean'):
        filter = Filter(type='affine',method=self.smooth,type_matrix=self.type_matrix, Nframe=self.Nframe)
        self.smoothPath =  filter.smooth_Trajectory(self.cameraPath)
        return self.smoothPath


    def warpingVideo(self):
        self.vid = cv2.VideoCapture(self.src_path)
        
        if self.traject_method == 'similarity':
            for idx in range(self.Nframe-1):
                checkFrame,img = self.vid.read()
                if not checkFrame:
                    break
                img = cv2.resize(img,self.imgSize)
                newX = self.orgTransform[idx,0] + self.smoothPath[idx,0] - self.cameraPath[idx,0]
                newY = self.orgTransform[idx,1] + self.smoothPath[idx,1] - self.cameraPath[idx,1]
                newA = self.orgTransform[idx,2] + self.smoothPath[idx,2] - self.cameraPath[idx,2]
                newS = self.orgTransform[idx,3] * self.smoothPath[idx,3] / self.cameraPath[idx,3]

                H = np.array(([np.cos(newA)*newS, -np.sin(newA)*newS, newX] ,
                                  [np.sin(newA)*newS,  np.cos(newA)*newS, newY]))

                newImg = cv2.warpAffine(img,H,self.imgSize)
                cv2.imshow('frame',newImg)
                self.saveVid.write(newImg)
                cv2.waitKey(20)

            self.saveVid.release()
            self.vid.release()


        elif (self.traject_method == 'affine'):
            for idx in range(self.Nframe-1):
                checkFrame,img = self.vid.read()
                if not checkFrame:
                    break
                img = cv2.resize(img,self.imgSize)
                A = np.matmul(self.orgTransform[idx,0:2,0:2] , np.matmul(self.smoothPath[idx,0:2,0:2] , inv(self.cameraPath[idx,0:2,0:2])))
                t = self.orgTransform[idx, 0:2, 2:3] + self.smoothPath[idx, 0:2, 2:3] - self.cameraPath[idx, 0:2, 2:3]
                H = np.hstack((A,t))

                newImg = cv2.warpAffine(img,H,self.imgSize)
                cv2.imshow('frame',newImg)
                self.saveVid.write(newImg)
                cv2.waitKey(20)
            self.saveVid.release()
            self.vid.release()


        else:
            for idx in range(self.Nframe-1):
                checkFrame,img = self.vid.read()
                if not checkFrame:
                    break
                img = cv2.resize(img,self.imgSize)
                h,w,c = img.shape
                (cx,cy) = (w//2,h//2)
                #A = np.array([[1,0],[0,1]])
                A = np.matmul(self.orgTransform[idx,0:2,0:2] , np.matmul(self.smoothPath[idx,0:2,0:2] , inv(self.cameraPath[idx,0:2,0:2])))
                t = self.orgTransform[idx, 0:2, 2:3] + self.smoothPath[idx, 0:2, 2:3] - self.cameraPath[idx, 0:2, 2:3]
                H = np.hstack((A,t))

                theta = np.zeros([3,3])
                theta[0,2]= (H[0,2] - H[0,0] - H[0,1] + 1)*w/2
                theta[1,2]= (H[1,2] - H[1,0] - H[1,1] + 1)*h/2
                theta[0,1] = H[0,1]*w/h
                theta[1,0] = H[1,0]*h/w
                theta[0,0] = H[0,0]
                theta[1,1] = H[1,1]
                theta[2,2] = 1.
                param = np.linalg.inv(theta)
                param = param[0:2,0:3]
               

                newImg = cv2.warpAffine(img,param,self.imgSize)
                cv2.imshow('frame',newImg)
                self.saveVid.write(newImg)
                cv2.waitKey(20)
            self.saveVid.release()
            self.vid.release()


        # elif self.traject_method == 'cnn_2':
        #     for idx in range(self.Nframe-1):
        #         checkFrame,img = self.vid.read()
        #         if not checkFrame:
        #             break
        #         img = cv2.resize(img,self.imgSize)
        #         h,w,c = img.shape
        #         image_batch = torch.from_numpy(img.astype(float))
        #         image_batch = image_batch.permute(2,0,1).unsqueeze(0)

        #         A = np.matmul(self.orgTransform[idx,0:2,0:2] , np.matmul(self.smoothPath[idx,0:2,0:2] , inv(self.cameraPath[idx,0:2,0:2])))

        #         #A = np.array([[1,0],[0,1]])
        #         t = self.orgTransform[idx, 0:2, 2:3] + self.smoothPath[idx, 0:2, 2:3] - self.cameraPath[idx, 0:2, 2:3]
        #         H = torch.from_numpy(np.hstack((A,t))).unsqueeze(0)

        #         out_size = torch.Size((1,c,h,w))
        #         sampling_grid = F.affine_grid(H,out_size,align_corners=True)
        #         warped_image_batch = F.grid_sample(image_batch.float(),sampling_grid.float(),align_corners=True)

        #         newImg = warped_image_batch[0,:,:,:]
        #         newImg = newImg.permute(1,2,0).numpy()

        #         cv2.imshow('frame',newImg.astype(np.uint8))
        #         self.saveVid.write(newImg.astype(np.uint8))
        #         cv2.waitKey(20)
        #     self.saveVid.release()
        #     self.vid.release()




















