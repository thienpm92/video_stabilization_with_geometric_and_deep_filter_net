import torch
from torchvision import transforms
from torch.autograd import Variable

class NormalizeImageDict(object):
    """
    Normalize image in dictionary
    normalize range is True, the image is divided by 255
    """
    def __init__(self,image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])

    def __call__(self,sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0
            sample[key] = self.normalize(sample[key])
        return sample