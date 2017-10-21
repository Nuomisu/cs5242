import numpy as np

def random_flips(X):
    out = None
    N, C, H, W = X.shape
    mask = 1
    out = np.zeros_like(X)
    out[mask==1] = X[mask==1,:,:,::-1]
    return out

def random_contrast(X, scale=(0.8, 1.2)):
    low, high = scale
    N = X.shape[0]
    out = np.zeros_like(X)
    scalar = np.random.uniform(low, high)
    out = X * scalar
    return out

def random_tint(X, scale=(-10, 10)):
    low, high = scale
    N, C = X.shape[:2]
    out = np.zeros_like(X)
    l = (scale[1]-scale[0])*np.random.random_sample((N,C))+scale[0]
    out = X+l[:,:,None,None]
    return out