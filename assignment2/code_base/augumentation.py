import numpy as np

def random_flips(X):
  out = None
  N, C, H, W = X.shape
  mask = 1
  out = np.zeros_like(X)
  out[mask==1] = X[mask==1,:,:,::-1]
  return out