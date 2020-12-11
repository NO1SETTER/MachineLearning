import csv
import numpy as  np

def sigmoid(x):
    if x>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))

def logistic_function(x):
    return .5 * (1 + np.tanh(.5 * x))


def train(train_data, train_label):
    """
    Input:
        train_data: np.array with shape [N, 17]. Input.
        train_label: np.array with shape [N, 1]. Label.
    Return:
        beta: np.array with shape [1, 17]. Optimal params with gradient descent method
    """
    N = train_data.shape[0]
    a = 0.0000066
    beta = np.ones((1, 17)) *0.44
    for i in range(1000):
        #[N,1]
        c=logistic_function(np.dot(train_data,beta.T))
        #[N,1]
        b=c-train_label
        #[17,1]
        err=np.dot(train_data.T,b)
        beta = beta-a*err.T
    return beta

def test(test_data,test_label,params):
    '''
    :param test_data: np.array with shape [N,17].
    :param test_label: np.array with shape [N,1]
    :param params:each element is an np.array with shape [1,17]
    :return:
    '''
    right=0
    total=0
    TP=np.zeros(26)
    FP=np.zeros(26)
    TN=np.zeros(26)
    FN=np.zeros(26)
    precision=np.zeros(26)
    recall=np.zeros(26)
    for k in range(len(test_data)):
        flags=[]
        for p in params:
            flag=logistic_function(np.dot(np.mat(test_data[k]),p.T))
            flags.append(float(flag))
        cls=-1
        key=-1
        for i in range(len(flags)):
            nowkey=flags[i]
            if nowkey>key:
                key=nowkey
                cls=i+1
        total+=1
        if cls==test_label[k]:
            right+=1
            TP[cls-1]+=1
            for i in range(26):
                if i!=cls-1:
                    TN[i]+=1
        else:
            FP[cls-1]+=1
            FN[int(test_label[k]-1)]+=1
    for i in range(26):
        precision[i]=TP[i]/float(TP[i]+FP[i])
        recall[i]=TP[i]/float(TP[i]+FN[i])
    precision_ma=precision.sum()/26
    recall_ma=recall.sum()/26
    F_ma=2*precision_ma*recall_ma/(precision_ma+recall_ma)
    precision_mi=TP.sum()/(TP.sum()+FP.sum())
    recall_mi=TP.sum()/(TP.sum()+FN.sum())
    F_mi=2*precision_mi*recall_mi/(precision_mi+recall_mi)
    print("Acurracy:"+str(right/total))
    print("micro Precision:"+str(precision_mi))
    print("micro Recall:"+str(recall_mi))
    print("micro F1:"+str(F_mi))
    print("macro Precision:"+str(precision_ma))
    print("macro Recall:"+str(recall_ma))
    print("macro F1:"+str(F_ma))
    return float(right/total)


with open('train_set.csv','r') as t1:
    train_data=np.array(np.loadtxt(t1,dtype=np.float, delimiter=",",skiprows=1))
with open('test_set.csv','r') as t2:
    test_data=np.array(np.loadtxt(t2,dtype=np.float, delimiter=",",skiprows=1))
    train_label=[]
    test_label=[]
    for row in train_data:
        train_label.append(row[16])
        row[16]=1

    for row in test_data:
        test_label.append(row[16])
        row[16]=1

    params=[[]]
    for i in range(1,27):
        train_label1=[]
        for ele in train_label:
            if ele == i:
                train_label1.append(1)
            else:
                train_label1.append(0)
        param=train(train_data,np.mat(train_label1).T)
        params.append(param)
    params=np.array(params)
    params=params[1:len(params)]
    test(test_data,np.mat(test_label).T,params)

