
# coding: utf-8

# # XGBoost Parametre Tuning for Rental Listing Inquiries 

# 第三周作业 在 Rental Listing Inquiries 数据上练习 xgboost 参数调优
# 数据说明:
#     
# Rental Listing Inquiries 数据集是 Kaggle 平台上的一个分类竞赛任务,需要根据
# 公寓的特征来预测其受欢迎程度(用户感兴趣程度分为高、中、低三类)。其
# 中房屋的特征 x 共有 14 维,响应值 y 为用户对该公寓的感兴趣程度。评价标准
# 为 logloss。
# 
# 数据链接:https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries
# 
# 为减轻大家对特征工程的入手难度,以及统一标准,数据请用课程网站提供的
# 特征工程编码后的数据(RentListingInquries_FE_train.csv)或稀疏编码的形式
# (RentListingInquries_FE_train.bin)。
# 
# xgboost 既可以单独调用,也可以在sklearn 框架下调用。
# 
# 大家可以随意选择。若采用 xgboost 单独调用使用方式,建议读取稀疏格式文件。
# 
# 关于特征工程的过程,可参看文件:FE_RentListingInqueries.ipynb
# 
# 作业要求:
# 采用 xgboost 模型完成商品分类(需进行参数调优)

# ## 第一步： 直接调用xgboost内嵌的cv寻找最佳的参数n_estimators

# 首先 import 必要的模块

# In[1]:


from xgboost import XGBClassifier # 通过scikit-learn调用XGBoost
import xgboost as xgb # 直接调用XGBoost

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from matplotlib import pyplot
import seaborn as sns
get_ipython().run_line_magic('matplotlid', 'inline')


# ## 读取数据

# In[2]:


# path to where the data lies
dpath = './data/'
train = pd.read_csv(dpath +"RentListingInquries_FE_train.csv")
test = pd.read_csv(dpath +"RentListingInquries_FE_test.csv")
#train.head()


# ## Variable Identification

# In[3]:


train.info()


# In[4]:


test.info()


# In[5]:


# drop outcome
y_train = train['interest_level']
X_train = train.drop(["interest_level"], axis=1)
X_test = test


# In[6]:


# prepare cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)


# 默认参数，此时学习率为0.1，比较大，观察弱分类数目的大致范围 （采用默认参数配置，看看模型是过拟合还是欠拟合）

# In[7]:


#直接调用xgboost内嵌的交叉验证（cv），可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证
def modelfit(alg, X_train, y_train, cv_folds=None, early_stopping_rounds=10):
    xgb_param = alg.get_xgb_params()
    xgb_param['num_class'] = 9
    
    #直接调用xgboost，而非sklarn的wrapper类
    xgtrain = xgb.DMatrix(X_train, label = y_train)
        
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
             metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
  
    cvresult.to_csv('1_nestimators.csv', index_label = 'n_estimators')
    
    #最佳参数n_estimators
    n_estimators = cvresult.shape[0]
    
    # 采用交叉验证得到的最佳参数n_estimators，训练模型
    alg.set_params(n_estimators = n_estimators)
    alg.fit(X_train, y_train, eval_metric='mlogloss')  
    
    #Predict training set:
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train, train_predprob)
    print("logloss of train set :",logloss)
    print("最佳参数n_estimators :",n_estimators)


# In[8]:


#params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}
xgb1 = XGBClassifier(
        learning_rate =0.1, # 学习率
        n_estimators=1000,  #数值大没关系，cv会自动返回合适的n_estimators (树的个数)
        max_depth=5,
        min_child_weight=1, 
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        objective= 'multi:softprob',
        seed=3)

modelfit(xgb1, X_train, y_train, cv_folds = kfold)


# In[9]:


cvresult = pd.DataFrame.from_csv('1_nestimators.csv')
test_means = cvresult['test-mlogloss-mean']
test_stds = cvresult['test-mlogloss-std'] 
        
train_means = cvresult['train-mlogloss-mean']
train_stds = cvresult['train-mlogloss-std'] 

x_axis = range(0, cvresult.shape[0])
        
pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
pyplot.title("XGBoost n_estimators vs Log Loss")
pyplot.xlabel( 'n_estimators' )
pyplot.ylabel( 'Log Loss' )
pyplot.savefig( 'n_estimators4_1.png' )

pyplot.show()


# In[10]:



cvresult = cvresult.iloc[100:]
# plot
test_means = cvresult['test-mlogloss-mean']
test_stds = cvresult['test-mlogloss-std'] 
        
train_means = cvresult['train-mlogloss-mean']
train_stds = cvresult['train-mlogloss-std'] 

x_axis = range(100,cvresult.shape[0]+100)
        
fig = pyplot.figure(figsize=(10, 10), dpi=100)
pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
pyplot.title("XGBoost n_estimators vs Log Loss")
pyplot.xlabel( 'n_estimators' )
pyplot.ylabel( 'Log Loss' )
pyplot.savefig( 'n_estimators_detail.png' )

pyplot.show()


# In[11]:


#生成提交测试结果
y_test_pred = xgb1.predict_proba(X_test)


# In[12]:


test_Id = pd.read_csv(dpath + 'test_Id.csv') 
y = pd.DataFrame(data = y_test_pred, columns = ['low','medium','high'])
df = pd.concat([test_Id, y], axis = 1)
df.to_csv('submission.csv')


# In[13]:


df.info()

