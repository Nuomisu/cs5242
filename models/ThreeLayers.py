import numpy as np 

from func import forward, forward_affine, backward, backward_affine

class ThreeLayers(object):
    
    
    def __init__(self, input_dim=14, hiddenL1 = 100, hiddenL2 = 14, num_output=4):
        self.W = [] # weights
        self.B = [] # biases

        self.hiddenL1 = hiddenL1
        self.hiddenL2 = hiddenL2

        # Init weights and biases

        self.B.append(np.zeros(hiddenL1))
        self.W.append(np.random.rand(input_dim, hiddenL1)*0.0001)

        self.B.append(np.zeros(hiddenL2))
        self.W.append(np.random.rand(hiddenL1, hiddenL2)*0.0001)
        
        self.B.append(np.zeros(num_output))
        self.W.append(np.random.rand(hiddenL2, num_output)*0.0001)
        

    def compute(self, X, Y):
        layers = []
        caches = []

        ## Forward calculation and active
        t_x = X
        for i in xrange(len(self.W)):
            t_w = self.W[i]
            t_b = self.B[i]

            if i == len(self.W)-1:
                t_layer, t_cache = forward_affine(t_x, t_w, t_b)
                layers.append(t_layer)  
                caches.append(t_cache)
            else:
                t_layer, t_cache = forward(t_x, t_w, t_b)
                t_x = t_layer
                layers.append(t_layer)
                caches.append(t_cache)

        if Y is None
            return layers[len(layers)-1]

        loss = 0
        gradiens_W = []
        gradiens_B = []

        ## Cross entropy 
        loss, de_loss = cross_entropy(layers[len(layers)-1], Y)

        ## Back propogation
        
        dout = de_loss

        for i in xrange len(cache) - 1:
            index = len(caches) - 1 - i
            t_cache = caches[index]
            
            if index == len(caches) - 1: 
                ## the last layer
                t_dx, t_dw, t_db = backward_affine(dout, t_cache)
                gradiens_W.append(t_dw)
                gradiens_B.append(t_db)
                dout = t_dx
            else:
                ## normal layer
                t_dx, t_dw, t_db = backward(dout, t_cache)
                gradiens_W.append(t_dw)
                gradiens_B.append(t_db)
                dout = t_dx

        return loss, gradiens_W, gradiens_B

