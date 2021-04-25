import torch
from torch.autograd import Variable
import numpy as np 
from geotnf.grid_gen import TpsGridGen



class PointTnf(object):
    def __init__(self,tps_grid_size=3, tps_reg_factor=0, use_cuda=True):
        self.use_cuda = use_cuda
        self.tpsTnf = TpsGridGen(grid_size=tps_grid_size,
                                 reg_factor = tps_reg_factor,
                                 use_cuda = self.use_cuda)

    def tpsPointTnf(self,theta,points):
        #point are expected in [B,2,N] where first row is X and 2nd row is Y
        #reshape point for applying Tps transform
        points = points.unsqueeze(3).transpose(1,3)
        #apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta, points)
        #undo reshaping
        warped_points = warped_points.transpose(3,1).squeeze(3)
        return warped_points


    def affPointTnf(self,theta,points):
        theta_mat = theta.view(-1,2,3)
        warped_points = torch.bmm(theta_mat[:,:,:2].float(),points)
        warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)     
        return warped_points