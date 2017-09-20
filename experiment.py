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
        self.loss_log = []

    def _reset(self):
        self.loss_log = []
        self.train_acc_log = []
        self.val_acc_log = []

    
    def train(self):

        for i in xrange(self.num_iterations):
            loss = self._iteration()

            train_acc = self.check_accuracy(self.x_train, self.y_train)
            test_acc = self.check_accuracy(self.x_test, self.y_test)
#            print "iteration %d, loss %f" %(i, loss)
            print '[Iteration %d / %d] loss: %f; Training Accuracy: %f; Test Accuracy: %f' % (i + 1, self.num_iterations, self.loss_log[-1], train_acc, test_acc)

    def _iteration(self):
        loss, gradiens_W, gradiens_B = self.model.compute(self.x_train, self.y_train)
        self.loss_log.append(loss)
        w_size = len(self.model.W)
        for i in xrange(len(self.model.W)):
            w_old = self.model.W[i]
            dw = gradiens_W[w_size - i - 1]
            b_old = self.model.B[i]
            db = gradiens_B[w_size - i - 1]
            
            next_w, next_learning_rate = sgd(w_old, dw, self.learning_rate)
            next_b, next_learning_rate = sgd(b_old, db, self.learning_rate)
            
            self.model.W[i] = next_w
            self.model.B[i] = next_b
            self.learning_rate = next_learning_rate
        #print self.model.W
        #print self.model.B
        return loss
            

    def check_accuracy(self, X, Y):
        y_pred = self.model.compute(X)
        right = 0
        for i in xrange(len(Y)):
            if y_pred[i] == Y[i]:
                right += 1
        acc = right *1.0 / len(Y)
        return acc





