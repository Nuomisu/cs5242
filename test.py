import numpy as np 
import func
numOfRecord = 10
dim = 1
hidden = 3

X = np.random.randn(numOfRecord, dim)
Y = np.zeros(numOfRecord)

W = np.random.randn(dim,hidden)
B = np.random.randn(hidden)

print W
print B
print X

result, cache = func.forward_affine(X, W, B)
print result
print cache

#print '[Iteration %d / %d] loss: %f; Training Accuracy: %f; Test Accuracy: %f-----------------' % (1, 4, 1.11, 2.22, 3.3)
