import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer


def AdaBoost(train_attr, train_label, rounds):
    rounds = 30
    sampleNum = len(train_label)
    D = []
    allTree = []
    Alpha = []
    for round in range(rounds):
        w = np.zeros(sampleNum,dtype=float)
        if round == 0:
            for i in range(sampleNum):
                w[i] = 1 / sampleNum
        else:
            w = D[round - 1]
        decTree = DecisionTreeClassifier(max_depth=3)
        decTree.fit(train_attr, train_label, sample_weight=w)
        pred = decTree.predict(train_attr)
        error = float(0)
        for i in range(sampleNum):
            if pred[i] != train_label[i]:
                error = error + w[i]
        if error > 0.5:
            break
        alpha = 0.5 * np.log((1 - error) / error)
        for i in range(sampleNum):
            w[i] = w[i] * np.exp(-alpha * pred[i] * train_label[i])
        allTree.append(decTree)
        Alpha.append(alpha)
        D.append(w)
    return allTree, Alpha


def CalculateAUC(test_attr, test_label, trees, Alpha):
    weighted_pred = np.zeros(len(test_label), dtype=float)
    for i in range(len(trees)):
        nowTree = trees[i]
        pred = nowTree.predict(test_attr)  # 第i个树对所有样本的预测
        alpha = Alpha[i]  # 第i个树的权重
        for j in range(len(test_label)):
            weighted_pred[j] += pred[j]*alpha

    for i in range(len(test_label)):
        if weighted_pred[i] < 0:
            weighted_pred[i] = -1
        else:
            weighted_pred[i] = 1
    rightpreds = 0
    for i in range(len(weighted_pred)):
        if weighted_pred[i] == test_label[i]:
            rightpreds += 1
    correct_rate = rightpreds / len(weighted_pred)
    #print("TrainCorrectRate:" + str(correct_rate))
    fpr, tpr, thresholds = roc_curve(weighted_pred, test_label)
    return auc(fpr, tpr), correct_rate


def TestOnTestSet(test_attr, test_label, trees, Alpha):
    weighted_pred = np.zeros(len(test_label), dtype=float)
    for i in range(len(trees)):
        now_tree = trees[i]
        pred = now_tree.predict(test_attr)  # 第i个树对所有样本的预测
        alpha = Alpha[i]  # 第i个树的权重
        for j in range(len(test_label)):
            weighted_pred[j] += pred[j]*alpha

    for i in range(len(test_label)):
        if weighted_pred[i] < 0:
            weighted_pred[i] = -1
        else:
            weighted_pred[i] = 1
    rightpreds = 0
    for i in range(len(weighted_pred)):
        if weighted_pred[i] == test_label[i]:
            rightpreds += 1
    correct_rate = rightpreds/len(weighted_pred)
    print("TestCorrectRate:"+str(correct_rate))
    return correct_rate


def CrossValidation(train_attr, train_label, test_attr, test_label):
    rounds = 50
    auc_val = 0
    test_correct_rate = 0
    train_correct_rate = 0
    kf = KFold(5)
    folds = 0
    for train_index, test_index in kf.split(train_attr):
        print('------------------------------------')
        print("Fold:"+str(folds+1))
        x_train, x_test = train_attr[train_index], train_attr[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        trees, weights = AdaBoost(x_train, y_train, rounds)
        ret1, ret2 = CalculateAUC(x_test, y_test, trees, weights)
        auc_val += ret1
        test_correct_rate += ret2
        folds = folds + 1
        train_correct_rate += TestOnTestSet(test_attr, test_label, trees, weights)
        print("auc_val="+str(auc_val/folds))
    print('------------------------------------')
    print('Sum:')
    print('Best BaseLearner Number:'+str(rounds))
    print('Auc:'+str(auc_val/5))
    #print('TrainCorrectRate:'+str(test_correct_rate/5))
    print('TestCorrectTate:'+str(train_correct_rate/5))



train_data = []
test_data = []
with open('./data/adult_data.csv') as t1:
    reader = csv.reader(t1)
    for row in reader:
        train_data.append(row)

with open('./data/adult_test.csv') as t2:
    reader = csv.reader(t2)

    for row in reader:
        test_data.append(row)


train_data = np.array(train_data)
test_data = np.array(test_data)
union_data = np.concatenate((test_data,train_data))

OE = OrdinalEncoder()
OE.fit(union_data)


train_data = OE.transform(train_data)
test_data = OE.transform(test_data)  # 大无语事件,测试数据和训练数据字符串都不一样
union_data = OE.transform(union_data)
# label 0,1 <= 划为负例, 2,3 > 划为正例

# 有一些为0的数值代表空值需要先把它转为nan再把它填充为众数
has_empty_cols = [1, 6, 13]
for i in range(len(train_data)):
    for j in has_empty_cols:
        if train_data[i, j] == 0:
            train_data[i, j] = np.nan
for i in range(len(test_data)):
    for j in has_empty_cols:
        if test_data[i, j] == 0:
            test_data[i, j] = np.nan

# 空值填充
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(union_data)
train_data = imp.transform(train_data)
test_data = imp.transform(test_data)

# 处理上面出现的大无语事件
for i in range(len(train_data)):
    if train_data[i][14] == 0 or train_data[i][14] == 1:
        train_data[i][14] = -1
    else:
        train_data[i][14] = 1

for i in range(len(test_data)):
    if test_data[i][14] == 0 or test_data[i][14] == 1:
        test_data[i][14] = -1
    else:
        test_data[i][14] = 1

train_attr = train_data[:, 0:-1]
train_label = train_data[:, -1]
test_attr = test_data[:, 0:-1]
test_label = test_data[:, -1]

CrossValidation(train_attr, train_label, test_attr, test_label)