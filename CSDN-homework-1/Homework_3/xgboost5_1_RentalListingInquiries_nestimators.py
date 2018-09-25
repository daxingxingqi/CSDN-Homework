
# coding: utf-8

# # XGBoost Parametre Tuning for Rental Listing Inquiries

# # 第五步：调整正则化参数：reg_alpha 和reg_lambda

# In[1]:


from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd 
import numpy as np

import math

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from matplotlib import pyplot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# path to where the data lies
dpath = './data/'
train = pd.read_csv(dpath +"RentListingInquries_FE_train.csv")
test = pd.read_csv(dpath +"RentListingInquries_FE_test.csv")
#train.head()


# In[3]:


# drop outcome
y_train = train['interest_level']
X_train = train.drop(["interest_level"], axis=1)
X_test = test


# In[4]:


# prepare cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)


# In[5]:


reg_alpha = [ 1.5, 2]    #default = 0, 测试0.1,1，1.5，2
reg_lambda = [0.5, 1, 2]      #default = 1，测试0.1， 0.5， 1，2

param_test5_1 = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)
param_test5_1


# In[6]:


xgb5_1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=267,  #第二轮参数调整得到的n_estimators最优值
        max_depth=5,
        min_child_weight=5,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.9,
        colsample_bylevel = 0.7,
        objective= 'multi:softprob',
        seed=3)


gsearch5_1 = GridSearchCV(xgb5_1, param_grid = param_test5_1, scoring='neg_log_loss',n_jobs=-1, cv=kfold)
gsearch5_1.fit(X_train , y_train)

gsearch5_1.grid_scores_, gsearch5_1.best_params_,     gsearch5_1.best_score_


# In[7]:


# summarize results
print("Best: %f using %s" % (gsearch5_1.best_score_, gsearch5_1.best_params_))
test_means = gsearch5_1.cv_results_[ 'mean_test_score' ]
test_stds = gsearch5_1.cv_results_[ 'std_test_score' ]
train_means = gsearch5_1.cv_results_[ 'mean_train_score' ]
train_stds = gsearch5_1.cv_results_[ 'std_train_score' ]

pd.DataFrame(gsearch5_1.cv_results_).to_csv('my_preds_reg_alpha_reg_lambda_1.csv')

# plot results
test_scores = np.array(test_means).reshape(len(reg_alpha), len(reg_lambda))
train_scores = np.array(train_means).reshape(len(reg_alpha), len(reg_lambda))

#log_reg_alpha = [0,0,0,0]
#for index in range(len(reg_alpha)):
#   log_reg_alpha[index] = math.log10(reg_alpha[index])
    
for i, value in enumerate(reg_alpha):
    pyplot.plot(reg_lambda, -test_scores[i], label= 'reg_alpha:'   + str(value))
#for i, value in enumerate(min_child_weight):
#    pyplot.plot(max_depth, train_scores[i], label= 'train_min_child_weight:'   + str(value))
    
pyplot.legend()
pyplot.xlabel( 'reg_alpha' )                                                                                                      
pyplot.ylabel( '-Log Loss' )
pyplot.savefig( 'reg_alpha_vs_reg_lambda1.png' )


# In[9]:


best_model = gsearch5_1.best_estimator_


# In[10]:


#生成提交测试结果
y_test_pred = best_model.predict_proba(X_test)
test_Id = pd.read_csv(dpath + 'test_Id.csv') 
y = pd.DataFrame(data = y_test_pred, columns = ['low','medium','high'])
df = pd.concat([test_Id, y], axis = 1)
df.to_csv('submission_final.csv')


# In[11]:


df.info()

