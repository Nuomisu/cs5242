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
            x_train.append(np.array(row))
    with open(path+'/x_test.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            x_test.append(np.array(row))
    with open(path+'/y_train.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            y_train.append(np.array(row))
    with open(path+'/y_test.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            y_test.append(np.array(row))
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = readfile("data")
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape