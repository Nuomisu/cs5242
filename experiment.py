import numpy as np 

from func import sgd, adam

class Experiment(object):

    def __init__(self, model, data, num_iterations, learning_rate):
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_test = data['x_test']
        self.y_test = data['y_test']
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def _reset(self):
        self.loss_log = []
        self.train_acc_log = []
        self.val_acc_log = []

    
    def train(self):

        for i in xrange(self.num_iterations):
            self._iteration()

            train_acc = self.check_accuracy(self.x_train, self.y_train)
            test_acc = self.check_accuracy(self.x_test, self.y_test)

            print '[Iteration %d / %d] loss: %f; Training Accuracy: %f; Test Accuracy: %f' % (i + 1, num_iterations, self.loss_log[-1], train_acc, test_acc)



    def _iteration(self):


        loss, gradiens_W, gradiens_B = self.model.compute(self.x_train, self.y_train)
        self.loss_log.append(loss)
        
        for i in xrange(self.model.W):
            w_old = self.model.W[i]
            dw = gradiens_W[i]
            b_old = self.model.B[i]
            dw = gradiens_B[i]
            
            next_w, next_learning_rate = sgd(w_old, dw, self.learning_rate)
            next_w, next_learning_rate = sgd(b_old, dw, self.learning_rate)
            
            self.model.W[i] = next_w
            self.model.B[i] = next_w
            self.learning_rate = next_learning_rate
            

    def check_accuracy(self, X, Y):
        
        y_pred = self.model.loss(X)
        
        acc = np.mean(y_pred == Y)

    return acc





