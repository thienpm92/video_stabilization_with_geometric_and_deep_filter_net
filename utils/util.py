import torch



def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))