# import numpy as np
# import sys , os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# from layer_net import Two_layer_net
# import matplotlib as plt

# network = Two_layer_net(input_size=784,hidden_size=100,output_size=10)

# (x_train,t_train),(x_test,t_test) = load_mnist(one_hot_label = True , normalize= True)

# iters_num = 10000
# batch_size = 100
# train_size = x_train.shape[0]
# learning_rate = 0.1

# train_loss_list = []

# for _ in range(iters_num):
#     batch_mask = np.random.choice(train_size,batch_size)
#     x_batch = x_train[batch_size]
#     t_batch = t_train[batch_size]
    
#     grad = network.numerical_grad(x_batch,t_batch)
    
#     for keys in ('W1','b1','W2','b2'):
#         network.params[keys]-=learning_rate*grad[keys]
    
#     loss = network.loss(x_batch,t_batch)
#     train_loss_list.append(loss)

    
    
    
# print(train_loss_list)

# plt.plot(train_loss_list)
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.show()

    
# 훈련 데이터로 손실 함수 값을 확인하지 않고 시험 데이터로 기록

import numpy as np
import sys , os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from layer_net import Two_layer_net
import matplotlib as plt

network = Two_layer_net(input_size=784,hidden_size=100,output_size=10)

(x_train,t_train),(x_test,t_test) = load_mnist(one_hot_label = True , normalize= True)

iters_num = 10000
batch_size = 100
train_size = x_train.shape[0]
learning_rate = 0.1

iter_per_epoch = max(1,train_size/batch_size)

train_loss_list = []
train_acc_list=[]
test_acc_list=[]

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_size]
    t_batch = t_train[batch_size]
    
    grad = network.numerical_grad(x_batch,t_batch)
    
    for keys in ('W1','b1','W2','b2'):
        network.params[keys]-=learning_rate*grad[keys]
    
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc ,test_acc :"+str(train_acc)+","+str(test_acc))
    
    


plt.plot(train_loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
