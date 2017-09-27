import csv
import sys
import numpy as np
import models.ThreeLayers as three
from experiment import Experiment
from plot import print_graph

def readfile(path):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    x_a= []
    y_a= []
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
    with open(path+'/x_a.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            newrow = np.array(row).astype(np.float)
            x_a.append(newrow)
    with open(path+'/y_a.csv', 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            newrow = np.array(row).astype(np.int)
            y_a.append(newrow)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_a), np.array(y_a)

def loadwb(wpath, bpath):
    w = []
    b = []
    with open(wpath, 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            w.append(row[1:])
    with open(bpath, 'rb') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            b.append(row[1:])
    return w,b

def q2_123(data):
    case = int(sys.argv[1])
    model = three.ThreeLayers(case)
    exp = Experiment(model, data, 100, 1)
    exp.train()
    print_graph(exp.loss_log)

def q2_4(data, path):
    case = int(sys.argv[1])
    wfile = "w-100-40-4.csv"
    bfile = "b-100-40-4.csv"
    if case == 2:
        wfile = "w-28-6-4.csv"
        bfile = "b-28-6-4.csv"
    elif case == 3:
        wfile = "w-14-28-4.csv"
        bfile = "b-14-28-4.csv"
    w, b = loadwb(path+wfile, path+bfile)
    model = three.ThreeLayers(case, 14, 4, w, b)
    data["x_train"] = data["x_a"]
    data["y_train"] = data["y_a"]
    exp = Experiment(model, data, 1, 1)
    w,b = exp.check_speicalpoint()
    # output dw db
    with open("result/"+wfile, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for wouter in w:
            for line in wouter:
                writer.writerow(line)
    with open("result/"+bfile, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in b:
            writer.writerow(line)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_a, y_a = readfile("data")
    data = {"x_train": x_train, "y_train": y_train,
            "x_test": x_test, "y_test": y_test,
            "x_a": x_a, "y_a": y_a}
    q2_123(data)
#    q2_4(data, "data/c/") 

