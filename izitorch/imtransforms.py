"""
Set of simple image transformations, similar to what can be found in torchvision but that work with mutlispectral data
(more than 3 or 4 channels)
"""

from torch import tensor
import numpy as np
from scipy import ndimage


def to_array(t):
    return t.numpy()


def to_tensor(a):
    return tensor(a)


def horizontal_flip(a):
    """Flips a (C,H,W) image horizontally"""
    return np.flip(a, axis=0).copy()  # to avoid negative stride problem when casting to tensor


def vertical_flip(a):
    """Flips a (C,H,W) image vertically"""
    return np.flip(a, axis=1).copy()


def random_rotation(a, angle=180, nrange=100):
    """Randomly rotates a (H,W,C) image. The angle is radomly chosen between -angle and + angle"""
    if not type(angle) == list:
        angle = [i for i in range(-angle, angle, (2 * angle) // nrange)]
    alpha = np.random.choice(angle)

    return ndimage.rotate(a, angle=alpha, axes=(1, 2), reshape=False)


def rotate(angle):
    return lambda x: random_rotation(x, angle=angle)


def space_shuffle(a):
    """Randomly shuffles the pixels of an image
    input: image tensor CxHxW (channel first)"""

    out = a[:, np.random.permutation(range(a.shape[1])), :]
    out = out[:, :, np.random.permutation(range(out.shape[2]))]

    return out


def random_sequence_rotation(a, angle=180, nrange=100):
    """ Rotates a tensor along the two last dimensions (H,W).
    Should be used to rotate each element of an image sequence using the same angle.
    """
    if not type(angle) == list:
        angle = [i for i in range(-angle, angle, (2 * angle) // nrange)]
    alpha = np.random.choice(angle)

    return ndimage.rotate(a, angle=alpha, axes=(-2, -1), reshape=False, order=1)


def rotate_sequence(angle):
    return lambda x: random_sequence_rotation(x, angle=angle)


def add_jitter(P):
    sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
    P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P
