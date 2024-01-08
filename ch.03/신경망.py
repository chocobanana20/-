import numpy as np
import sys , os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle
from common.function import sigmoid , softmax

def get_data():
    (x_train , t_train) , (x_test,t_test) = load_mnist(normalize = True , flatten = True , one_hot_label = False)
    
    return x_test, t_test 

# network가 있어야 됨 . 근데 처음이라 pickle 파일에서 가져온 듯
def init_network():
    with open("sample_weight.pkl",'rb') as f: # rb는 뭘까
        network = pickle.load(f)
    
    return network


# 입력층의 뉴런이 784 (이미지 크기) , 2개의 은닉층을 거쳐 출력층의 개수는 10개 (0부터 9 구분)
def predict(network , x): # network에 get_data로 얻은 데이터 가져와서 넣기 . x에는 x_test가 들어갈 껀데 이게 행렬임
    W1 , W2 , W3 = network['W1'],network['W2'],network['W3']
    b1 , b2 ,b3 = network['b1'],network['b2'],network['b3'] # 딕셔너리로 저장
    
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = softmax(a3)
    
    return y # 0부터 9 까지의 확률 



x_test, t_test = get_data()
network = init_network()
accuracy_cnt=0

# for i in range(len(x_test)):
#     y=predict(network,x_test[i])
#     if np.argmax(y)==t_test[i]:  # np.argmax로 0부터 9 까지 수 중에 가장 높은 확률로 나올 것이라 예측한 것을 찾아야됨
        
#         accuracy_cnt+=1
        
        
# 배치처리 구현
batch_size = 100
for i in range(0,len(x_test),batch_size):
    y = predict(network,x_test[i:i+batch_size]) # 결과가 2차원 배열 axis =1 해서 각각
    p = np.argmax(y,axis= 1)
    accuracy_cnt += np.sum(p==t_test[i:i+batch_size])


print("Accuracy:"+str(float(accuracy_cnt)/len(x_test)))
