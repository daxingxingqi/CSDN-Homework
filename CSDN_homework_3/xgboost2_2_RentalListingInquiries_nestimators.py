
# coding: utf-8

# # XGBoost Parametre Tuning for Rental Listing Inquiries

# # 第二步：调整树的参数：max_depth & min_child_weight
# (粗调，参数的步长为2；下一步是在粗调最佳参数周围，将步长降为1，进行精细调整)

# In[1]:


from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd 
import numpy as np

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


#max_depth 建议3-10， min_child_weight=1／sqrt(ratio_rare_event) =5.5
max_depth = [4,5,6]
min_child_weight = [4,5,6]
param_test2_2 = dict(max_depth=max_depth, min_child_weight=min_child_weight)
param_test2_2


# In[6]:


xgb2_2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=267,  #第一轮参数调整得到的n_estimators最优值
        max_depth=5,
        min_child_weight=5,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel = 0.7,
        objective= 'multi:softprob',
        seed=3)


gsearch2_2 = GridSearchCV(xgb2_2, param_grid = param_test2_2, scoring='neg_log_loss',n_jobs=-1, cv=kfold)
gsearch2_2.fit(X_train , y_train)

gsearch2_2.grid_scores_, gsearch2_2.best_params_,     gsearch2_2.best_score_


# In[7]:


gsearch2_2.cv_results_


# In[8]:


# summarize results
print("Best: %f using %s" % (gsearch2_2.best_score_, gsearch2_2.best_params_))
test_means = gsearch2_2.cv_results_[ 'mean_test_score' ]
test_stds = gsearch2_2.cv_results_[ 'std_test_score' ]
train_means = gsearch2_2.cv_results_[ 'mean_train_score' ]
train_stds = gsearch2_2.cv_results_[ 'std_train_score' ]

pd.DataFrame(gsearch2_2.cv_results_).to_csv('my_preds_maxdepth_min_child_weights_2.csv')

# plot results
test_scores = np.array(test_means).reshape(len(min_child_weight), len(max_depth))
train_scores = np.array(train_means).reshape(len(min_child_weight), len(max_depth))

for i, value in enumerate(min_child_weight):
    pyplot.plot(max_depth, test_scores[i], label= 'test_min_child_weight:'   + str(value))
#for i, value in enumerate(min_child_weight):
#    pyplot.plot(max_depth, train_scores[i], label= 'train_min_child_weight:'   + str(value))
    
pyplot.legend()
pyplot.xlabel( 'max_depth' )                                                                                                      
pyplot.ylabel( '- Log Loss' )
pyplot.savefig( 'max_depth_vs_min_child_weght2.png' )

