import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable
# PHUONG THUC RESIZE VA WARPING KHAC BAN GOC
class SynthDataset(Dataset):
    def __init__(self,  dataset_image_path,
                        output_size = (480,640),
                        geometric_model='affine',
                        dataset_size=0,
                        transform=None,
                        random_t = 0.5,
                        random_s = 0.5,
                        random_alpha = 1/6,
                        random_t_tps = 0.4,
                        four_point_hom=True):

        self.out_h, self.out_w = output_size
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.four_point_hom = four_point_hom
        self.dataset_size = dataset_size
        self.dataset_image_path= dataset_image_path
        self.lst = os.listdir(dataset_image_path)
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    # def transform(self,image):
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Resize(size=self.output_size),
    #                                     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    #     image /= 255.0
    #     image  = transform(image)
    #     return image


    def __len__(self):
        return len(self.lst)

    def __getitem__(self,index):
        #image = Image.open(self.dataset_image_path+self.lst[index])
        image = cv2.imread(self.dataset_image_path+self.lst[index],1)

        if self.geometric_model=='affine':
            rot_angle = (np.random.rand(1)-0.5)*2*np.pi/12  #between -np.pi/12 and np.pi/12
            sh_angle = (np.random.rand(1)-0.5)*2*np.pi/6    #between -np.pi/6 and np.pi/6
            lambda_1 = 1+(2*np.random.rand(1)-1)*0.05       #between 0.75 and 1.25
            lambda_2 = 1+(2*np.random.rand(1)-1)*0.1       #between 0.75 and 1.25
            tx = (2*np.random.rand(1)-1)*0.1               #between -0.25 and 0.25
            ty = (2*np.random.rand(1)-1)*0.1

            R_sh = np.array([[np.cos(sh_angle[0]),np.sin(sh_angle[0])],
                             [-np.sin(sh_angle[0]),np.cos(sh_angle[0])]])

            R_alpha = np.array([[np.cos(rot_angle[0]),-np.sin(rot_angle[0])],
                                [np.sin(rot_angle[0]),np.cos(rot_angle[0])]])

            D = np.diag([lambda_1[0],lambda_1[0]])
            A = R_alpha @ D
            #A = R_alpha @ R_sh.transpose() @ D @ R_sh

            theta = np.array([A[0,0],A[0,1],tx,A[1,0],A[1,1],ty],dtype=np.float32)

        if self.geometric_model=='hom':
            theta_hom = np.array([-1,-1,1,1,-1,1,-1,1])
            theta = theta_hom + (np.random.rand(8)-0.5)*2*self.random_t_tps

        theta = torch.from_numpy(theta.astype(float))
        #image = transforms.ToTensor()(image)
        
        image = torch.from_numpy(image.astype(float))
        image = image.permute(2,0,1).float()


        #permute order of img to C-H-W
        #image = image.transpose(1,2).transpose(0,1)

        #resize image using bilinear sampling with identity affine tnf
        if image.size()[0]!=self.out_h or image.size()[1]!=self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False)).data.squeeze(0)

        sample = {'image':image, 'theta':theta}
        if self.transform:
            sample = self.transform(sample)

        return sample
       

