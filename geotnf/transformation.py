import os
import torch
import numpy as np 
from torch.nn.modules.module import Module 
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from utils.util import expand_dim


class SynthPairTnf(object):
    """
    Generate a synthetically warped training pair using affine transform
    """
    def __init__(self,use_cuda=True,geometric_model='affine',crop_factor=9.0/16.0,output_size=(240,240),padding_factor=0.5,occlusion_factor=0):
        self.occlusion_factor=occlusion_factor
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h,self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine',out_h=self.out_h,out_w=self.out_w,use_cuda=self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model,out_h=self.out_h,out_w=self.out_w,use_cuda=self.use_cuda)

    def __call__(self,batch):
        image_batch,theta_batch = batch['image'],batch['theta']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()
        b,c,h,w = image_batch.size()
        #generate symmetric padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)
        #convert to variable
        image_batch = Variable(image_batch,requires_grad=False)
        theta_batch = Variable(theta_batch,requires_grad=False)
        #get crop image
        cropped_image_batch = self.rescalingTnf(image_batch=image_batch,
                                                theta_batch=None,
                                                padding_factor=self.padding_factor,
                                                crop_factor=self.crop_factor)   #Indentity is used as no theta give
        #get transformed image
        warped_image_batch = self.geometricTnf(image_batch=image_batch,
                                               theta_batch=theta_batch,
                                               padding_factor=self.padding_factor,
                                               crop_factor=self.crop_factor)
        if self.occlusion_factor!=0:
            rolled_indices_1 = torch.LongTensor(np.roll(np.arange(b),1))
            rolled_indices_2 = torch.LongTensor(np.roll(np.arange(b),2))
            mask_1 = self.get_occlusion_mask(cropped_image_batch.size(),self.occlusion_factor)
            mask_2 = self.get_occlusion_mask(cropped_image_batch.size(),self.occlusion_factor)
            
            if self.use_cuda:
                rolled_indices_1 = rolled_indices_1.cuda()
                rolled_indices_2 = rolled_indices_2.cuda()
                mask_1 = mask_1.cuda()
                mask_2 = mask_2.cuda()
            #apply mask
            cropped_image_batch = torch.mul(cropped_image_batch,1-mask_1) + torch.mul(cropped_image_batch[rolled_indices_1,:],mask_1)
            warped_image_batch = torch.mul(warped_image_batch,1-mask_2) + torch.mul(warped_image_batch[rolled_indices_1,:],mask_2)
        return {'source_image':cropped_image_batch,'target_image':warped_image_batch,'theta_GT':theta_batch}



    def symmetricImagePad(self,image_batch,padding_factor):
        b,c,h,w = image_batch.size()
        pad_h,pad_w = int(h*padding_factor),int(w*padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1))
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1))
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1))
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1))
        if self.use_cuda:
            idx_pad_left = idx_pad_left.cuda()
            idx_pad_right = idx_pad_right.cuda()
            idx_pad_top = idx_pad_top.cuda()
            idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,
                                 image_batch.index_select(3,idx_pad_right)),3)
        image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch,
                                 image_batch.index_select(2,idx_pad_bottom)),2)
        return image_batch

    def get_occlusion_mask(self,mask_size,occlusion_factor):
        b,c,out_h,out_w = mask_size
        #create mask of occluded portion
        box_w = torch.round(out_w*torch.sqrt(torch.FloatTensor([occlusion_factor]))*(1+(torch.rand(b)-0.5)*2/5))
        box_h = torch.round(out_h*out_w*occlusion_factor/box_w)
        box_x = torch.floor(torch.rand(b)*(out_w-box_w))
        box_y = torch.floor(torch.rand(b)*(out_h-box_h))
        box_w = box_w.int()
        box_h = box_h.int()
        box_x = box_x.int()
        box_y = box_y.int()
        mask = torch.zeros(mask_size)
        for i in range(b):
            mask[i,:,box_y[i]:box_y[i]+box_h[i], box_x[i]:box_x[i]+box_w[i]]=1
        #convert to variable
        mask = Variable(mask)
        return mask


class GeometricTnf(object):
    """
    Geometric transformation to an image batch (warped in Pytorch variable)
    (can be used with no transformation to perform bilienar resizing)
    """
    def __init__(self,geometric_model='affine',tps_grid_size=3, tps_reg_factor=0, out_h=240, out_w=240, offset_factor=None, use_cuda=True):
        self.out_w = out_w
        self.out_h = out_h
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda
        self.offset_factor = offset_factor

        if geometric_model=='affine' and offset_factor is None:
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w,use_cuda=True)
        elif geometric_model=='affine' and offset_factor is not None:
            self.gridGen = AffineGridGenV2()
        elif geometric_model=='hom':
            print('not construct HomoGridGen')


        if offset_factor is not None:
            self.gridGen.grid_X = self.gridGen.grid_X/offset_factor
            self.gridGen.grid_Y = self.gridGen.grid_Y/offset_factor
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()


    def __call__(self,image_batch,theta_batch=None,out_h=None,out_w=None,return_warped_image=True, return_sampling_grid=False, padding_factor=1.0, crop_factor=1.0):
        if image_batch is None:
            b=1
        else:
            b = image_batch.size(0)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3).contiguous()
            theta_batch = Variable(theta_batch,requires_grad=False)
        #check if output dim have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h!=self.out_h or out_w!=self.out_w):
            if self.geometric_model=='affine':
                gridGen = AffineGridGen(out_h=out_h, out_w=out_w,use_cuda=True)
            elif self.geometric_model=='hom':
                print('not construct HomoGridGen')
        else:
            gridGen = self.gridGen
        sampling_grid = gridGen(theta_batch)

        #rescale gird accornding to crop factor and padding factor
        if padding_factor !=1 or crop_factor != 1:
            sampling_grid = sampling_grid*(padding_factor*crop_factor)
        #rescale grid according to offset factor
        if self.offset_factor is not None:
            sampling_grid = sampling_grid*self.offset_factor

        if return_sampling_grid and not return_warped_image:
            return sampling_grid
        #sample transformed image
        warped_image_batch = F.grid_sample(image_batch,sampling_grid.float(),align_corners=False)

        if return_sampling_grid and return_warped_image:
            return(warped_image_batch,sampling_grid)
        return warped_image_batch




class AffineGridGen(Module):
    def __init__(self,out_h=240,out_w=240,out_ch=3,use_cuda=True):
        super(AffineGridGen,self).__init__()
        self.out_w = out_w
        self.out_h = out_h
        self.out_ch = out_ch

    def forward(self,theta):
        b=theta.size()[0]
        if not theta.size()==(b,2,3):
            theta = theta.view(-1,2,3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta,out_size,align_corners=False)

class AffineGridGenV2(Module):
    def __init__(self,out_h=240,out_w=240,use_cuda=True):
        super(AffineGridGenV2,self).__init__()
        self.out_w = out_w
        self.out_h = out_h
        self.use_cuda = use_cuda

        #create grid in numpy
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X,requires_grad=False)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self,theta):
        b=theta.size()[0]
        if not theta.size==(b,6):
            theta = theta.view(b,6)
            theta = theta.contiguous()
        t0 = theta[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1 = theta[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2 = theta[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3 = theta[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4 = theta[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5 = theta[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X,0,b)
        grid_Y = expand_dim(self.grid_Y,0,b)
        grid_Xp = grid_X*t0 + grid_Y*t1 +t2
        grid_Yp = grid_X*t3 + grid_Y*t4 +t5
        return torch.cat((grid_Xp,grid_Yp),3)
