# 손실 함수



# import numpy as np

# # 1 . 오차 제곱합
# def sum_squres_error(y,t):
#     return 0.5 * np.sum((y-t)**2)  # y , t 가 신경망의 출력 , 정답 레이블

# # 2 . 교차 엔트로피 오차

def cross_entropy_error(y,t):
    delta = 1e-7 # 아주 작은 값을 더해줘서 음의 무한대로 가지 않도록
    return -np.sum(t*np.log(y+delta))

# 미니배치 10개씩 랜덤하게 추출해서 실행해보기


# import sys , os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist

# (x_train , t_train),(x_test,t_test) = load_mnist(normalize = True , flatten = True , one_hot_label = True)
# train_size = x_train[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size . batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# def numberical_diff(f,x):
#     h = 1e-4
#     return (f(x+h)-f(x-h))/(2*h)

# def numberical_gradient(f,x):
#     h = 1e-4
#     grad = np.zeros_like(x)
    
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         grad[idx] = (f(tmp_val+h)-f(tmp_val-h))/(2*h)
        
#     return grad

# def grad_descent(f,init_x,lr=0.01,step_num=100):
#     x = init_x
#     for _ in range(step_num):
#         grad = numberical_gradient(f,x)
#         x = x - lr*grad[x]  # 
        
#     return x

# import sys , os 
# sys.path.append(os.pardir)
# from common.function import softmax , cross_entropy_error
# import numpy as np
# from common.gradient import numerical_gradient

# class Simple_net:
#     def __init__(self):
#         self.W = np.random.randn(2,3)
    
#     def predict(self,x):
#         return np.dot(x,self.W)
    
#     def loss(self,x,t):
#         z= self.predict(x)
#         y = softmax(z)
#         loss = cross_entropy_error(y,t)
        
#         return loss
    
    
# net = Simple_net()
# print(net.W)
# x = np.array([0.6,0.9])
# p = net.predict(x)
# print(p)
# t = np.array([0,0,1])
# loss = net.loss(x,t)
# print(loss)


import sys , os
sys.path.append(os.pardir)
from common.function import *
from common.gradient import numerical_gradient
import numpy as np

class Two_layer_net:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        
        
    def predict(self,x):
        a1 = np.dot(x,self.params['W1'])+self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1,self.params['W2'])+self.params['b2']
        y = softmax(a2)
        
        return y
    
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        return np.sum(y==t)/float(x.shape[0])

    def numerical_grad(self,x,t):
        loss_W = lambda W : self.loss(x,t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        
        return grads






