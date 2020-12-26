import csv
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer


def RandomForest(train_attr, train_label):
    samplenum = len(train_label)
    rounds = 20
    trees = []
    for round in range(rounds):
        temp_train_attr = []
        temp_train_label = []
        for i in range(samplenum):
            row = random.randint(0, samplenum-1)
            temp_train_attr.append(train_attr[row])
            temp_train_label.append(train_label[row])
        dectree = DecisionTreeClassifier(max_depth=10, max_features='log2')
        dectree.fit(temp_train_attr, temp_train_label)
        trees.append(dectree)
    return trees


def CalculateAUC(test_attr, test_label, trees):
    samplenum = len(test_label)
    vote_result = np.zeros(samplenum)
    for t in trees:
        pred = t.predict(test_attr)
        for i in range(samplenum):
            vote_result[i] = vote_result[i] + pred[i]
    for i in range(samplenum):
        if vote_result[i] > 0:
            vote_result[i] = 1
        elif vote_result[i] < 0:
            vote_result[i] = -1
        else:
            vote_result[i] = random.choice([-1,1])

    rightpreds = 0
    for i in range(samplenum):
        if vote_result[i] == test_label[i]:
            rightpreds += 1
    print("TrainCorrectTate:"+str(rightpreds/samplenum))
    fpr, tpr, threshold = roc_curve(vote_result, test_label)
    return  auc(fpr,tpr)


def TestOnTestSet(test_attr, test_label, trees):
    samplenum = len(test_label)
    vote_result = np.zeros(samplenum)
    for t in trees:
        pred = t.predict(test_attr)
        for i in range(samplenum):
            vote_result[i] = vote_result[i] + pred[i]
    for i in range(samplenum):
        if vote_result[i] > 0:
            vote_result[i] = 1
        elif vote_result[i] < 0:
            vote_result[i] = -1
        else:
            vote_result[i] = random.choice([-1,1])

    rightpreds = 0
    for i in range(samplenum):
        if vote_result[i] == test_label[i]:
            rightpreds += 1
    print("TrainCorrectTate:"+str(rightpreds/samplenum))


def CrossValidation(train_attr, train_label, test_attr, test_label):
    auc_val = 0
    kf = KFold(5)
    folds = 0
    for train_index, test_index in kf.split(train_attr):
        print("Fold:"+str(folds+1))
        x_train, x_test = train_attr[train_index], train_attr[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        trees = RandomForest(x_train, y_train)
        auc_val += CalculateAUC(x_test, y_test, trees)
        folds = folds + 1
        TestOnTestSet(test_attr, test_label, trees)
        print("auc_val="+str(auc_val/folds))


train_data = []
test_data = []
with open('adult_data.csv') as t1:
    reader = csv.reader(t1)
    for row in reader:
        train_data.append(row)

with open('adult_test.csv') as t2:
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


CrossValidation(train_attr,train_label,test_attr,test_label)