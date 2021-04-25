import os
import numpy as np 
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable


class TpsGridGen(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen,self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        self.reg_factor = reg_factor

        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w), np.linspace(-1,1,out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X,requires_grad=False)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y, P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1))    #size (N,1)
            P_Y = np.reshape(P_Y,(-1,1))
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = Variable(self.compute_L_inverse(P_X,P_Y).unsqueeze(0), requires_grad=False)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_X = Variable(self.P_X,requires_grad=False)
            self.P_Y = Variable(self.P_Y,requires_grad=False)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self,theta):
        warped_grid = self.apply_transform(theta,torch.cat((self.grid_X,self.grid_Y),3))
        return warped_grid




    def compute_L_inverse(self,X,Y):
        N = X.size()[0] #num of points
        #construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2) + torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0] = 1 #make diagonal 1 to avoid Nan in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        if self.reg_factor !=0:
            K += torch.eye(K.size(0),K.size(1)*self.reg_factor)
        #construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1), torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transform(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        #point shoud be in the [B,H,W,2]
        #where points[:,:,:,0] are the X coords and [:,:,:,1] are the Y coords 

        #input are corresponding control points P_i
        batch_size = theta.size()[0]
        #split theta into point coordinate
        Q_X = theta[:,:self.N,:,:].unsqueeze(3)
        Q_Y = theta[:,self.N:,:,:].unsqueeze(3)

        #get spatial dimension of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        #repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))

        #compute weights for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        #reshape
        #W_X,W_Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)

        #compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        #reshape
        #A_X,A_Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)

        #compute distance P_i - (grid_X,grid_Y)
        #grid is expanded in point dim 4, but not in batch dim 0, as points P_X, P_Y are fixed for all batch
        point_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        point_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))

        if points_b==1:
            delta_X = point_X_for_summation - P_X
            delta_Y = point_Y_for_summation - P_Y
        else:
            #use expanded P_X,P_Y in batch dimension
            delta_X = point_X_for_summation - P_X.expand_as(point_X_for_summation)
            delta_Y = point_Y_for_summation - P_Y.expand_as(point_Y_for_summation)

        dist_squared = torch.pow(delta_X,2) + torch.pow(delta_Y,2)
        #U: size [1,H,W,1,N]
        U = torch.mul(dist_squared,torch.log(dist_squared))

        #expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])

        points_X_prime = A_X[:,:,:,:,0] + \
                         torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                         torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)),4)
        points_Y_prime = A_Y[:,:,:,:,0] + \
                         torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                         torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)),4)

        return torch.cat((points_X_prime,points_Y_prime),3)

