import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from geotnf.transformation import GeometricTnf




def symmetricImagePad(image_batch,padding_factor=0.5):
    b,c,h,w = image_batch.size()
    pad_h,pad_w = int(h*padding_factor),int(w*padding_factor)
    idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1))
    idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1))
    idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1))
    idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1))

    image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,
                             image_batch.index_select(3,idx_pad_right)),3)
    image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch,
                             image_batch.index_select(2,idx_pad_bottom)),2)
    return image_batch



def generate_data(path='',savepath='',modeData='positive',geometric_mode='affine',output_size=(240,240)):
    lst = os.listdir(path+'/')
    out_h, out_w = output_size
    rescalingTnf = GeometricTnf(geometric_model=geometric_mode,out_h=out_h,out_w=out_w,use_cuda=False)
    geometricTnf = GeometricTnf(geometric_model=geometric_mode,out_h=out_h,out_w=out_w,use_cuda=False)
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])


    if modeData=='positive':
        label = 1
        count = 0
        for idx in lst:
            image = cv2.imread(path+'/%s'%idx,1)
            image = cv2.resize(image,(640,480))

            if geometric_mode=='affine':
                rot_angle = (np.random.rand(1)-0.5)*2*np.pi/12
                scale = 1+(2*np.random.rand(1)-1)*0.05
                tx = (2*np.random.rand(1)-1)*0.05
                ty = (2*np.random.rand(1)-1)*0.05
                R = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                              [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])
                D = np.diag([scale[0],scale[0]])
                A = R@D
                theta = np.array([A[0,0],A[0,1],tx,A[1,0],A[1,1],ty])

            theta = torch.from_numpy(theta.astype(float)).unsqueeze(0)

            image = torch.from_numpy(image.astype(float))
            image = image.permute(2,0,1).float()
            image /= 255.0
            image_batch = image.unsqueeze(0)
            #image_batch = normalize(image).unsqueeze(0)
            image_batch = symmetricImagePad(image_batch)
            cropped_image_batch = rescalingTnf( image_batch=image_batch,
                                                theta_batch=None,
                                                padding_factor=0.5,
                                                crop_factor=9.0/16.0) 
            warped_image_batch = geometricTnf( image_batch=image_batch,
                                                    theta_batch=theta,
                                                    padding_factor=0.5,
                                                    crop_factor=9.0/16.0)
            datum = (cropped_image_batch.squeeze(0).numpy(), warped_image_batch.squeeze(0).numpy(), theta.squeeze(0).numpy())
            np.save(savepath+'datum_'+'%s'%count,datum)
            count += 1
            if count == 5000:
                break

    elif modeData == 'negative':
        label = 0
        count = 50000
        numImage = len(lst)
        idx2 = 1
        theta_identity = np.array([[1,0,0,0,1,0]])
        for idx1 in range(numImage-1):
            image_1 = cv2.imread(path+'/%s'%lst[idx1],1)
            image_2 = cv2.imread(path+'/%s'%lst[idx2],1)

            image_1 = cv2.resize(image_1,(640,480))
            image_2 = cv2.resize(image_2,(640,480))

            image_1 = torch.from_numpy(image_1.astype(float))
            image_1 = image_1.permute(2,0,1).float()
            image_1 /= 255.0
            image_1 = normalize(image_1).unsqueeze(0)

            image_2 = torch.from_numpy(image_2.astype(float))
            image_2 = image_2.permute(2,0,1).float()
            image_2 /= 255.0
            image_2 = normalize(image_2).unsqueeze(0)

            cropped_image_1= rescalingTnf( image_batch=image_1,
                                                theta_batch=None,
                                                padding_factor=0.5,
                                                crop_factor=9.0/16.0)
            cropped_image_2= rescalingTnf( image_batch=image_2,
                                                theta_batch=None,
                                                padding_factor=0.5,
                                                crop_factor=9.0/16.0)
            datum = (cropped_image_1.squeeze(0).numpy(), cropped_image_2.squeeze(0).numpy(), 
                     theta_identity, label)
            np.save(savepath+'datum_'+'%s'%(count+10),datum)
            count += 1
            idx2 += 1
            if count == 60000:
                break
    print("finish")






def main():
    DataPath = '/media/vipl/DATA/dataset/MSCOCO2017/val2017'
    lst = os.listdir(DataPath+'/')
    SavePath = '/media/vipl/DATA/dataset/geometric_matching_dataset/valid/'
    print('positive')
    generate_data(path=DataPath,savepath=SavePath,modeData='positive',geometric_mode='affine',output_size=(240,240))
    # print('negative')
    # generate_data(path=DataPath,savepath=SavePath,modeData='negative',geometric_mode='affine',output_size=(240,240))
    print('finish')


if __name__=='__main__':
    main()