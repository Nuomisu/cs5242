import numpy as np 

from func import sgd, adam

class Experiment(object):

    def __init__(self, model, data, num_iterations, learning_rate, batchsize = 16):
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_test = data['x_test']
        self.y_test = data['y_test']
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.loss_log = []
        self.config_w = [None]*len(model.W)
        self.config_b = [None]*len(model.B)
        self.batchsize = batchsize

    def _reset(self):
        self.loss_log = []
        self.train_acc_log = []
        self.val_acc_log = []

    # for 2 (1,2,3) 
    def train(self):
        for i in xrange(self.num_iterations):
            loss = self._iteration()
            train_acc = self.check_accuracy(self.x_train, self.y_train)
            test_acc = self.check_accuracy(self.x_test, self.y_test)
#            print "iteration %d, loss %f" %(i, loss)
            print '[Iteration %d / %d] loss: %f; Training Accuracy: %f; Test Accuracy: %f' % (i + 1, self.num_iterations, loss, train_acc, test_acc)
            info = {"loss":loss, "train_acc": train_acc, "test_acc": test_acc}
            self.loss_log.append(info)

    # for 2 (1,2,3) 
    def _iteration(self):
        batch_num = (len(self.x_train)+self.batchsize-1)/self.batchsize
        for i in xrange(batch_num):
            batch_x = None
            batch_y = None
            if i == batch_num -1:
                batch_x = self.x_train[i*self.batchsize:len(self.x_train)]
                batch_y = self.y_train[i*self.batchsize:len(self.y_train)]
            else:
                batch_x = self.x_train[i*self.batchsize:(i+1)*self.batchsize]
                batch_y = self.y_train[i*self.batchsize:(i+1)*self.batchsize]
            loss, gradiens_W, gradiens_B = self.model.compute(batch_x, batch_y)
            w_size = len(self.model.W)
            for i in xrange(len(self.model.W)):
                w_old = self.model.W[i]
                dw = gradiens_W[w_size - i - 1]
                b_old = self.model.B[i]
                db = gradiens_B[w_size - i - 1]
                #next_w, next_learning_rate = sgd(w_old, dw, self.learning_rate)
                #next_b, next_learning_rate = sgd(b_old, db, self.learning_rate)
                #self.learning_rate = next_learning_rate
                next_w, self.config_w[i] = adam(w_old, dw, self.config_w[i])
                next_b, self.config_b[i] = adam(b_old, db, self.config_b[i])
                self.model.W[i] = next_w
                self.model.B[i] = next_b
            #print self.model.W
            #print self.model.B
        return loss
            
    # for 2 (1,2,3) 
    def check_accuracy(self, X, Y):
        y_pred = self.model.compute(X)
        right = 0
        for i in xrange(len(Y)):
            if y_pred[i] == Y[i]:
                right += 1
        acc = right *1.0 / len(Y)
        return acc

    # for 2 (4)
    def check_speicalpoint(self):
        w = []
        b = []
        w_size = len(self.model.W)
        loss, gradiens_W, gradiens_B = self.model.compute(self.x_train, self.y_train)
        for i in xrange(len(self.model.W)):
            dw = gradiens_W[w_size - i - 1]
            db = gradiens_B[w_size - i - 1]
            w.append(dw)
            b.append(db)
        return w,b


