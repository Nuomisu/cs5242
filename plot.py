import matplotlib.pyplot as plt

def print_graph(loss_log):
    x = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    for i, info in enumerate(loss_log):
        loss = info["loss"]
        train_acc = info["train_acc"]
        test_acc = info["test_acc"]
        test_loss = info["test_loss"]
        x.append(i+1)
        y.append(loss)
        y1.append(train_acc)
        y2.append(test_acc)
        y3.append(test_loss)
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
    plt.figure(4)
    plt.xlabel('iterations number')
    plt.ylabel('test loss')
    plt.title('test loss vs iterations for test data')
    plt.ylim([0,2])
    plt.xlim([1,len(loss_log)])
    plt.xticks(x)
    plt.plot(x, y3)
    plt.savefig('test_loss.png')
