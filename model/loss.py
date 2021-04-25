import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from geotnf.pointTnf import PointTnf


class TransformedGridLoss(nn.Module):
    def __init__(self,geometric_model='affine',use_cuda=True,grid_size=20):
        super(TransformedGridLoss,self).__init__()
        self.geometric_model = geometric_model
        #define virtual gird of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.PointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda()

    def forward(self,theta,theta_GT):
        #expand gird according to batch size
        batch_size= theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)

        #compute transformed grid points using estimated and GT tnfs
        if self.geometric_model=='affine':
            P_prime = self.PointTnf.affPointTnf(theta,P)
            P_prime_GT = self.PointTnf.affPointTnf(theta_GT,P)
        elif self.geometric_model=='hom':
            P_prime = self.PointTnf.homPointTnf(theta,P)
            P_prime_GT = self.PointTnf.homPointTnf(theta_GT,P)
        elif self.geometric_model=='tps':
            P_prime = self.PointTnf.tpsPointTnf(theta,P)
            P_prime_GT = self.PointTnf.tpsPointTnf(theta_GT,P)
        #compute MSE loss
        loss = torch.sum(torch.pow(P_prime-P_prime_GT,2),1)
        loss = torch.mean(loss)
        return loss