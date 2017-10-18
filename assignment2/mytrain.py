import numpy as np
import matplotlib.pyplot as plt
from code_base.classifiers.cnn import *
from code_base.data_utils import get_CIFAR2_data
from code_base.layers import *
from code_base.solver import Solver

data = get_CIFAR2_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

model = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
print('start train')
solver.train()
print('finish train')

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

