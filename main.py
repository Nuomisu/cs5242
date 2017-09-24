import csv
import sys
import numpy as np
import models.ThreeLayers as three
from experiment import Experiment
import matplotlib.pyplot as plt

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

def print_graph(loss_log):
    x = []
    y = []
    y1 = []
    y2 = []
    for i, info in enumerate(loss_log):
        loss = info["loss"]
        train_acc = info["train_acc"]
        test_acc = info["test_acc"]
        x.append(i+1)
        y.append(loss)
        y1.append(train_acc)
        y2.append(test_acc)
    plt.figure(1)
    plt.xlabel('iterations number')
    plt.ylabel('loss')
    plt.title('loss vs iterations for training data')
    plt.ylim([0,2])
    plt.xlim([1,len(loss_log)])
    plt.xticks(x)
    plt.plot(x, y)
    plt.savefig('loss.png')
    plt.figure(2)
    plt.xlabel('iterations number')
    plt.ylabel('train acc')
    plt.title('train acc vs iterations for training data')
    plt.ylim([0,1])
    plt.xlim([1,len(loss_log)])
    plt.xticks(x)
    plt.plot(x, y1)
    plt.savefig('train_acc.png')
    plt.figure(3)
    plt.xlabel('iterations number')
    plt.ylabel('test acc')
    plt.title('test acc vs iterations for test data')
    plt.ylim([0,1])
    plt.xlim([1,len(loss_log)])
    plt.xticks(x)
    plt.plot(x, y2)
    plt.savefig('test_acc.png')
    

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = readfile("data")
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    data = {"x_train": x_train, "y_train": y_train,
            "x_test": x_test, "y_test": y_test}
    case = int(sys.argv[1])
    model = three.ThreeLayers(case)
    exp = Experiment(model, data, 50, 1)
    exp.train()
    print_graph(exp.loss_log)
