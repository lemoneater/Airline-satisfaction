# -*- coding: utf-8 -*-
"""

@author: 张升
"""
# =============================================================================

# Data Source
# https://www.kaggle.com/johndddddd/customer-satisfaction

# Context
# US Airline passenger satisfaction survey
# 
# Content
# Satisfaction:Airline satisfaction level(satisfied, neutral or dissatisfaction)
# 
# Age:The actual age of the passengers
# 
# Gender:Gender of the passengers (Female, Male)
# 
# Type of Travel:Purpose of the flight of the passengers (Personal Travel, Business Travel)
# 
# Class:Travel class in the plane of the passengers (Business, Eco, Eco Plus)
# 
# Customer Type:The customer type (Loyal customer, disloyal customer)
# 
# Flight distance:The flight distance of this journey
# 
# Inflight wifi service:Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
# 
# Ease of Online booking:Satisfaction level of online booking
# 
# Inflight service:Satisfaction level of inflight service
# 
# Online boarding:Satisfaction level of online boarding
# 
# Inflight entertainment:Satisfaction level of inflight entertainment
# 
# Food and drink:Satisfaction level of Food and drink
# 
# Seat comfort:Satisfaction level of Seat comfort
# 
# On-board service:Satisfaction level of On-board service
# 
# Leg room service:Satisfaction level of Leg room service
# 
# Departure/Arrival time convenient:Satisfaction level of Departure/Arrival time convenient
# 
# Baggage handling:Satisfaction level of baggage handling
# 
# Gate location:Satisfaction level of Gate location
# 
# Cleanliness:Satisfaction level of Cleanliness
# 
# Check-in service:Satisfaction level of Check-in service
# 
# Departure Delay in Minutes:Minutes delayed when departure
# 
# Arrival Delay in Minutes:Minutes delayed when Arrival
# 
# =============================================================================

# In[]
import pandas as pd
import numpy as np
import os


sub_path=os.path.abspath(__file__).split("airline.py")[0]
# 数据读取
df=pd.read_csv(sub_path+"\satisfaction.csv")



# In[]
# part1:数据预处理
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# In[]
#分类数据编码

#需要编码的列
features=['Satisfaction','Gender','Customer Type','Type of Travel','Class'] 
le=LabelEncoder()
for i in features:
    df[i]=le.fit_transform(df[i])
    
# =============================================================================
# Satisfaction:Airline satisfaction level(1:Satisfaction, 0:neutral or dissatisfaction)
# Gender:Gender of the passengers (0:Female, 1:Male)
# Customer Type:The customer type (0:Loyal customer, 1:disloyal customer)
# Type of Travel:Purpose of the flight of the passengers (1:Personal Travel, 0:Business Travel)
# Class:Travel class in the plane of the passengers (0:Business, 1:Eco, 2:Eco Plus)
# =============================================================================


print(df.isnull().any()) #检查是否存在缺失值

# In[]
#缺失值处理处理
cols=df.columns.tolist()
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') 
#策略：mean:平均数，median:中位数，most_frequent:众数，constant:用fill_value的参数替换空值
#由于该列中数据分布不均匀，所以使用众数替代空值
temp=imp.fit_transform(df)
df=pd.DataFrame(temp,columns=cols)

print(df.isnull().any()) #再次检查是否存在缺失值

# In[]
# part2:数据可视化
import matplotlib.pyplot as plt
import seaborn as sns

# In[]
#客户类型和满意程度
sns.barplot(x='Customer Type', y='Satisfaction',data=df)
plt.show()
# Customer Type:The customer type (0:Loyal customer, 1:disloyal customer)

# In[]
#仓位和满意程度
sns.barplot(x='Class', y='Satisfaction',data=df)
plt.show()
# Class:Travel class in the plane of the passengers (0:Business, 1:Eco, 2:Eco Plus)

# In[]

#满意程度在飞行距离中的分布
sns.boxplot(x='Satisfaction', y='Flight Distance',hue='Class',data=df)
plt.show()

# In[]
#满意程度在出发延误时间中的分布
sns.boxplot(x='Satisfaction', y='Departure Delay in Minutes',hue='Type of Travel',data=df)
plt.ylim(0,200)
plt.show()



# In[]
#用相关系数矩阵查看出发延误和到达延误的相关性
corrmatrix1 = df.iloc[:,-2:].corr(method='pearson')  #pearson参数分析线性数据相关性（共线程度）
#散点图证明
sns.pointplot(x='Departure Delay in Minutes',y='Arrival Delay in Minutes',data=df)
#由于两个变量线性相关，且到达延误在处理前存在缺失值，所以只取Departure Delay in Minutes


# In[]
#将连续变量分箱方便进行卡方检验
#查看飞行距离分布
sns.boxplot( y='Flight Distance',data=df)
plt.show()
bins1=[0,1000,2000,3000,4000,df['Flight Distance'].argmax()]

#出发延误时间分布
sns.boxplot( y='Departure Delay in Minutes',data=df)
plt.show()
plt.ylim(0,60)
sns.boxplot( y='Departure Delay in Minutes',data=df)
plt.show()
bins2=[-1,15,60,120,240,800,df['Departure Delay in Minutes'].argmax()]


df['fl_dis_bins']=pd.cut(df['Flight Distance'],bins1,labels=False)
df['dpt_delay_bins']=pd.cut(df['Departure Delay in Minutes'],bins2,labels=False)
print(df.isnull().any()) #再次检查是否存在缺失值

# In[]
#用相关系数矩阵查看分类变量的相关性
corrmatrix2 = df.corr(method='spearman')  #kendall或spearman参数分析分类数据之间相关性
corr_check=corrmatrix2[np.abs(corrmatrix2) > 0.5]  # 选取相关系数绝对值大于0.5的变量
#选取这些变量进行下一步卡方检验
#'Gender','Customer Type','Food and drink','Inflight entertainment','Online support','On-board service','Leg room service','Checkin service'

# In[]
#根据卡方检验选取关联较大变量
import sklearn.feature_selection as feature_selection
df['Gender'] = df['Gender'].astype('int')
df['Customer Type'] = df['Customer Type'].astype('int')
chi2_test=feature_selection.chi2(df[['Gender','Customer Type','Food and drink',
                                    'Inflight entertainment','Online support',
                                    'On-board service','Leg room service',
                                    'Checkin service','fl_dis_bins','dpt_delay_bins']], df['Satisfaction'])#选取部分字段进行卡方检验




# In[]
# 划分训练集和测试集
import sklearn.model_selection as cross_validation

target = df['Satisfaction']  # 选取因变量
data=df.loc[:,['Gender','Customer Type','Food and drink',
                                    'Inflight entertainment','Online support',
                                    'On-board service','Leg room service',
                                    'Checkin service','fl_dis_bins','dpt_delay_bins']]  # 选取自变量

train_data, test_data, train_target, test_target = cross_validation.train_test_split(data,target, test_size=0.4, train_size=0.6 ,random_state=12345)
 # 划分训练集和测试集
 #随机数一定有种子，给出种子每次给出的随机划分是一样的，具有可验证性

# In[]
# part2:决策树建模
import sklearn.tree as tree
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics

#定义超参数搜索网格，最大深度和内部节点划分所需最小样本数
param_gridDT = {
     'max_depth':[10,11,12,13,14,15,16,17],
    'min_samples_split':[8,10,12,14,16,18,20] 
}

#进行网格搜索，采用交叉验证的方法，评价最优的超参数
DT = tree.DecisionTreeClassifier(criterion='entropy')
DTcv = GridSearchCV(estimator=DT, param_grid=param_gridDT, 
                   scoring='roc_auc', cv=4) 


#最优模型保存再DTcv之中，进行模型训练
DTcv.fit(train_data, train_target)

#获取最优参数
DTbest_params=DTcv.best_params_  

#获取模型评分
DTScore=DTcv.best_score_

# In[]
# 查看模型预测结果
train_est = DTcv.predict(train_data)  #  用模型预测训练集的结果
train_est_p=DTcv.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est=DTcv.predict(test_data)  #  用模型预测测试集的结果
test_est_p=DTcv.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率

#决策树对测试集分类准确度
DT_accuracy=DTcv.score(test_data, test_target)


# In[]
#绘制决策树roc曲线
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color='blue')
plt.plot(fpr_train, tpr_train, color='red')
plt.title("DT ROC")
plt.show()
#红色训练集，蓝色测试集



# In[]
#模型保存

import pydotplus
from IPython.display import Image
import pickle as pickle


DT = tree.DecisionTreeClassifier(criterion='entropy', max_depth=DTbest_params['max_depth'], min_samples_split=DTbest_params['min_samples_split']) # 当前支持计算信息增益和GINI
DT.fit(train_data, train_target)
dot_data = tree.export_graphviz(
    DT, 
    out_file=None, 
    feature_names=train_data.columns,
    max_depth=5,
    class_names=['0','1'],
    filled=True
) 


model_file = open(r'DT.model', 'wb')
pickle.dump(DT, model_file)
model_file.close()
            
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
with open('DT.png', 'wb') as f:
    f.write(graph.create_png())

# =============================================================================
# 模型读取
# model_load_file = open(r'DT.model', 'rb')
# model_load = pickle.load(model_load_file)
# model_load_file.close()
# 
# test_est_load = model_load.predict(test_data)
# pd.crosstab(test_est_load,test_est)
# =============================================================================


# In[]
#part3: 神经网络建模
from sklearn.neural_network import MLPClassifier

#定义搜索网格，隐藏层(层数，节点数)和L2正则化
param_gridMLP={'hidden_layer_sizes':[(50,),(100,),(100,100)],'alpha':[0.0001,0.001,0.01,0.1]}

#进行网格搜索，采用交叉验证的方法，评价最优的超参数
MLP = MLPClassifier(activation='logistic')
MLPcv = GridSearchCV(estimator=MLP, param_grid=param_gridMLP, 
                    scoring='roc_auc', cv=4)  


#神经网络最优模型保存在MLPcv中进行训练
MLPcv.fit(train_data,train_target) 

#获取最优参数
MLPbest_params=MLPcv.best_params_
  
#获取模型评分
MLPScore=MLPcv.best_score_

# In[]
# 查看神经网络模型预测结果
train_est1 = MLPcv.predict(train_data)  #  用模型预测训练集的结果
train_est_p1=MLPcv.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est1=MLPcv.predict(test_data)  #  用模型预测测试集的结果
test_est_p1=MLPcv.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率

#神经网络对测试集分类准确度
MLP_accuracy=MLPcv.score(test_data, test_target)


# In[]
#绘制神经网络roc曲线
fpr_test1, tpr_test1, th_test1 = metrics.roc_curve(test_target, test_est_p1)
fpr_train1, tpr_train1, th_train1 = metrics.roc_curve(train_target, train_est_p1)
plt.figure(figsize=[6,6])
plt.plot(fpr_test1, tpr_test1, color='blue')
plt.plot(fpr_train1, tpr_train1, color='red')
plt.title("MLP ROC")
plt.show()
#红色训练集，蓝色测试集



# =============================================================================
# # In[]
# #获取最优参数再次建模保存
# MLPbest_params=MLPcv.best_params_
# MLP = MLPClassifier(activation='logistic',solver=MLPbest_params['solver'],hidden_layer_sizes=MLPbest_params['hidden_layer_sizes'])
# MLP.fit(train_data, train_target)
# 
# model_file = open(r'MLP.model', 'wb')
# pickle.dump(MLP, model_file)
# model_file.close()
# =============================================================================


# In[]
#part4:模型比较

plt.figure(figsize=[6,6])
plt.plot(fpr_test1, tpr_test1, color='red')
plt.plot(fpr_test, tpr_test, color='blue')
plt.title("model comparison")
plt.show()
#两种模型在测试集上的拟合程度，红色神经网络，蓝色决策树
















