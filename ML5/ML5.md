# 机器学习作业5

## 181860152 周宇翔

[TOC]

### 1.[30pts]Naive Bayes Classifier

**(1)**
$$
Pr\{y=1|\pmb x=(1,1,0,1)\}=\frac{Pr\{(y,\pmb x)=(1,1,1,0,1)\}}{Pr\{\pmb x=(1,1,0,1)\}}=\frac{Pr\{\pmb x=(1,1,0,1)|y=1\}Pr\{y=1\}}{Pr\{\pmb x=(1,1,0,1)\}}\\
Pr\{y=0|\pmb x=(1,1,0,1)\}=\frac{Pr\{(y,\pmb x)=(0,1,1,0,1)\}}{Pr\{\pmb x=(1,1,0,1)\}}=\frac{Pr\{\pmb x=(1,1,0,1)|y=0\}Pr\{y=0\}}{Pr\{\pmb x=(1,1,0,1)\}}
$$
基于属性条件独立性假设
$$
Pr\{\pmb x=(1,1,0,1)|y=1\}Pr\{y=1\}=Pr\{x_1=1|y=1\}\Pr\{x_2=1|y=1\}Pr\{x_3=0|y=1\}\\Pr\{x_4=1|y=1\}Pr\{y=1\}=\frac{2}{3}*\frac{1}{3}*0*\frac{2}{3}*\frac{3}{5}=0\\
Pr\{\pmb x=(1,1,0,1)|y=0\}Pr\{y=0\}=Pr\{x_1=1|y=0\}\Pr\{x_2=1|y=0\}Pr\{x_3=0|y=0\}\\Pr\{x_4=1|y=0\}Pr\{y=0\}=(\frac{1}{2})^4*\frac{2}{5}=\frac{1}{40}
$$
由于$Pr\{y=1|\pmb x=(1,1,0,1)\}+Pr\{y=0|\pmb x=(1,1,0,1)\}=1$

计算可得$Pr\{y=1|\pmb x=(1,1,0,1)\}=0,Pr\{y=0|\pmb x=(1,1,0,1)\}=0$

**(2)**

使用Laplacian Correction后,上述的一些概率变为
$$
Pr\{x_1=1|y=1\}=\frac{3}{5},Pr\{x_2=1|y=1\}=\frac{2}{5},Pr\{x_3=0|y=1\}=\frac{1}{5},Pr\{x_4=1|y=1\}=\frac{3}{5}\\
Pr\{x_1=1|y=0\}=Pr\{x_2=1|y=0\}=Pr\{x_3=0|y=1\}=Pr\{x_4=1|y=1\}=\frac{1}{2}\\
Pr\{y=1\}=\frac{4}{7},Pr\{y=0\}=\frac{3}{7}
$$
于是有
$$
Pr\{\pmb x=(1,1,0,1)|y=1\}=\frac{3}{5}*\frac{2}{5}*\frac{1}{5}*\frac{3}{5}*\frac{4}{7}=\frac{72}{4375}\\
Pr\{\pmb x=(1,1,0,1)|y=0\}=(\frac{1}{2})^4*\frac{3}{7}=\frac{3}{112}
$$
根据概率之和为1以及两个概率成比例可知
$$
Pr\{y=1|\pmb x=(1,1,0,1)\}=\frac{384}{1009}=0.3806\\
Pr\{y=0|\pmb x=(1,1,0,1)\}=\frac{625}{1009}=0.6194
$$





### 2.[70pts]Ensemble Methods in Practice

#### (1)Code Realization

check AdaBoost.py RandomForest.py

#### (2)Pseudo Code For RandomForest Algorithm

```
RandomForest(D,A,k):  #DataSet,AttributeSet,k
	Let v be a node of the tree
	if all samples in D are of the same class
		label v as a leaf node
		return
	if A is empty or samples in D has the same value on A
		label v as a leaf node
		assign class of most samples in D to the class of v
		return
	if k<A
		select k attributes out of A,Let the attributes be A'
		select the best split attribute a out of A'
	else 
		select the best split attribute a out of A 
	for each attr on a
		generate a branch for v,Let D_u be the samples in D that have value u on attr		if D_u is empty
			label D_u as a leaf node
			assign class of most samples in D to the class of v
			return
		else 
			Let RandomForest(D_u,A\{attr},k)be a branch node
```

#### (3)Performances

在AdaBoost和RandomForest中,需要调整的参数主要是基学习器的数量,因此我们对基学习器数量在[10,50]范围内进行测试,即对于每个基学习器数都进行五折交叉验证,并计算在每折的平均AUC,最后选取最大平均AUC对应的基学习器数为最优基学习器数

测试结果见AdaBoostPerformance.txt,RandomForestPerformance.txt

得到结论如下:

+ AdaBoost(max_depth = 3)

>Best Base Learner Number :17
>Average Auc: 0.772355
>Correct Rate On TestSet:0.832676

测试10-50个学习器,发现在采用17个基学习器时能达到约0.77的AUC和在测试集83%的正确率,

+ RandomForest(max_depth = 10)

> Best Base Learner Number :23
> Average Auc: 0.811676
> Correct Rate On TestSet:0.851176

测试10-50个学习器,发现在采用17个基学习器时能达到约0.81的AUC和在测试集85%的正确率,当然由于算法随机的关系,最优的基学习器数仍会产生振荡.



#### (4)Sketches

AdaBoost 和 RandomForest算法虽然在思路上分别属于集成学习的两种类别,但是实现起来还是有很多相似处的,下面对每个阶段进行分析

##### 数据读入和处理

​	这次的数据集和之前的有所不同,之前的数据集的属性值均为离散的数字,且无空值.而这次提供的Adult数据集,不仅属性值存在整数,实数,字符串等多种问题,且存在空值,因此我们先要将数据集处理为一个可以供决策树使用的数据集

​	首先采用的是sklearn.preprocessing包里的OrdinalEncoder,它可以把数据集里的任意属性的字符串值按顺序转换成离散整数值,我们先提供给它所有训练及测试数据让它适配(fit),再用它分别对训练数据和测试数据进行transform即可.需要额外处理的是标签数据,因为OrdinalEncoder提供的是从0开始的离散值,而我们需要把标签转化为-1/1(另外吐槽一下训练数据和测试数据里同一属性值还出现了">50K"和">=50K."两种表示方式导致工作量增加)

​	把所有属性值都转化为整数之后,我们还需要进行空值填充,同样在sklearn.impute里提供了SimpleImputer这样的工具,这里采用众数填充即可.需要注意的是要先把会出现空值的属性中的空值('?')在OrdinalEncoder替换后的值(也就是0)替换成其它值(我选取的是np.nan),然后在Imputer调用时设置missing_value为这个值



##### 5折交叉验证

​	这一部分利用KFold包去自动生成所有训练数据的"五折",在每一折上利用在训练数据上划分出的训练集(4/5)训练模型,在训练数据上划分出的测试集(1/5)计算AUC,然后把计算出的模型应用在全部测试数据上计算测试集正确率,基于每一折的表现统计平均AUC,以衡量算法的性能.然后根据平均AUC的数量选取最佳的基学习器数量



##### 集成学习

###### AdaBoost

​	AdaBoost算法基本按照书8.3复刻,也没有太多的细节.大概思路就是初始设置所有样本权重相等,用数据集和权重训练出决策树,然后用该决策树(限制深度)预测一遍,对预测错误的样本加大权重,预测正确的样本减小权重,并根据该决策树的错误率计算该决策树在整个AdaBoost中的权重.然后用新的权重和数据集重复此过程,最后得到所有的决策树(基学习器)的集合以及它们相应的权重

###### RandomForest

​	RandomForest算法实现相对简单,我们只要做到在每一轮训练基学习器时,随机选取训练集即可,这里我自己用random包模拟了一个有放回采样的过程,然后把采样到的数据用于训练基学习器.另外RandomForest算法要求在决策树的每一层先随机选取一些属性再从这些属性中选取最优划分属性,这可以通过显示DecisionTreeClassifier的max_features参数来非常方便的实现



##### 计算AUC

​	训练出所有的基学习器后,无论是AdaBoost还是RandomForest都需要把所有基学习器的预测结果转化为最终唯一的预测结果.AdaBoost采用的方法是对于任何样本,把每个基学习器对它的预测结果进行加权求和,然后根据求和结果,大于0则判定为正例,小于0则判定为负例.RandomForest中不存在权重的问题,所以它更接近于把所有基学习器的结果近似于一个投票过程,最后选取票数更多的为预测结果.

​	得到我们的模型的预测结果pred后,我们只需要利用sklearn.metrics 中的auc包和roc_curve包,提供给它pred和真正的标签y即可得到AUC



##### 测试集测试

​	和计算AUC的过程基本一样,得到预测结果pred后直接和真实标签做比较即可计算正确率,在此就不作赘述

