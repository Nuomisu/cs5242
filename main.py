import csv
import sys
import numpy as np
import models.ThreeLayers as three
from experiment import Experiment

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
    exp = Experiment(model, data, 1000, 1)
    exp.train()

