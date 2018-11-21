"""
Set of simple image transformations, similar to what can be found in torchvision but that work with mutlispectral data
"""

from torch import tensor
import numpy as np
from scipy import ndimage





def to_array(t):
    return t.numpy()


def to_tensor(a):
    return tensor(a)


def horizontal_flip(a):
    return np.flip(a, axis=1).copy() #to avoid negative stride problem when casting to tensor


def vertical_flip(a):
    return np.flip(a, axis=0).copy()


def random_rotation(a,angle=180,nrange=100):
    if not type(angle) == list:
        angle = [i for i in range(-angle,angle,(2*angle)//nrange)]
    alpha = np.random.choice(angle)

    return ndimage.rotate(a,angle=alpha,axes=(1,2),reshape=False)

def rotate(angle):
    return lambda x: random_rotation(x,angle=angle)

def add_jitter(P):
    sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
    P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P