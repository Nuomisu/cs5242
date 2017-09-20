import csv
import numpy as np

def readfile(path):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    with open(path+'/x_train.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            newrow = np.array(row).astype(np.float)
            x_train.append(newrow)
    with open(path+'/x_test.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            newrow = np.array(row).astype(np.float)
            x_test.append(newrow)
    with open(path+'/y_train.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            newrow = np.array(row).astype(np.int)
            y_train.append(newrow)
    with open(path+'/y_test.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            newrow = np.array(row).astype(np.int)
            y_test.append(newrow)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def train(X, y):
    # initialize parameters randomly
    D = 14
    K = 4
    h = 100 # size of hidden layer
    W = 0.01 * np.random.randn(D,h)
    b = np.zeros((1,h))
    W2 = 0.01 * np.random.randn(h,K)
    b2 = np.zeros((1,K))
    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3 # regularization strength
    # gradient descent loop
    num_examples = X.shape[0]
    for i in xrange(10000):
        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2
        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        # compute the loss: average cross-entropy loss and regularization
        # corect_logprobs = -np.log(probs[np.arange(num_examples),y])
        logprobs = []
        for i in xrange(num_examples):
            logprobs.append(probs[i, y[i]])
        corect_logprobs = -np.log(np.array(logprobs))
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        print "iteration %d: loss %f" % (i, loss)
        # compute the gradient on scores
        dscores = probs
        for i in xrange(num_examples):
            dscores[i,y[i]] -= 1
        dscores /= num_examples
        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)
        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W
        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = readfile("data")
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    train(x_train, y_train) 
