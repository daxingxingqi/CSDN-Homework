
# coding: utf-8

# # 1. Import Necessary Tool Packages

# In[416]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

import matplotlib.pyplot as plt
import seaborn as sns
# color = sns.color_palette()

get_ipython().run_line_magic('mathplotlib', 'inline # generate figure inside console')


# # 2. Data explore

# ## 2.1 Import the data

# In[417]:


# path to where the data lies
dpath =  '/home/zichen/Desktop/CSDN/week1/data/'
data = pd.read_csv(dpath + "day.csv")
data.head()


# ## 2.2 Basic data information

# In[418]:


data.info()


# In[419]:


### Check is there are any nulls
data.isnull().sum()


# In[420]:


## statistic characteritics
data.describe()


# In[421]:


data.shape


# ## 2.3 Initial whole data explore

# In[422]:


# Split the train and test data
train = data[data.yr == 0] # train data
test = data[data.yr == 1]  # test data

# temp histogram/distribution
fig = plt.figure()
sns.distplot(data.temp.values, bins = 30, kde = False)
plt.xlabel('Temperature', fontsize = 12)
plt.show()


# In[423]:


# atemp histogram/distribution
fig = plt.figure()
sns.distplot(data.atemp.values, bins = 30, kde = False)
plt.xlabel('Sensible Temperature', fontsize = 12)
plt.show()


# In[424]:


# hum histogram/distribution
fig = plt.figure()
sns.distplot(data.hum.values, bins = 30, kde = False)
plt.xlabel('Humid', fontsize = 12)
plt.show()


# In[425]:


# y(casual) histogram/distribution
fig = plt.figure()
sns.distplot(data.casual.values, bins = 30, kde = True)
plt.xlabel('Casual users of sharing bikes', fontsize = 12)
plt.show()
plt.scatter(range(data.shape[0]),data["casual"].values,color='purple')
plt.title("Distribution of Casual users")


# In[426]:


# y(redistered) histogram/distribution
fig = plt.figure()
sns.distplot(data.registered.values, bins = 30, kde = True)
plt.xlabel('Registered users of sharing bikes', fontsize = 12)
plt.show()
plt.scatter(range(data.shape[0]),data["registered"].values,color='purple')
plt.title("Distribution of Registerd  users")


# In[427]:


# y(cnt) histogram/distribution
fig = plt.figure()
sns.distplot(data.cnt.values, bins = 30, kde = True)
plt.xlabel('CNT of sharing bikes', fontsize = 12)
plt.show()
plt.scatter(range(data.shape[0]),data["cnt"].values,color='purple')
plt.title("Distribution of CNT")


# ## 2.4 Data preparing

# In[428]:


# X_train and y_train from original data (month data may be saved)
y_train = train['cnt'].values
X_train = train.drop(['cnt', 'registered', 'casual', 'instant', 'dteday', 'season', 'yr', 'mnth'], axis = 1 )

# X_test and y_test from original data
y_test = test['cnt'].values
X_test = test.drop(['cnt', 'registered', 'casual', 'instant', 'dteday', 'season', 'yr', 'mnth'], axis = 1 )

# Data split for onehotencoder (categorical and numerical data)
c_X_train =  X_train.drop(['temp', 'atemp', 'hum', 'windspeed'], axis = 1)
n_X_train =  X_train.drop(['holiday', 'weekday', 'workingday','weathersit'], axis = 1)

c_X_test =  X_test.drop(['temp', 'atemp', 'hum', 'windspeed'], axis = 1)
n_X_test =  X_test.drop(['holiday', 'weekday', 'workingday','weathersit'], axis = 1)


# In[429]:


# Import  normalization packages
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
ss_X =  StandardScaler()
ss_y =  StandardScaler()

# Onehotencoder
ohe = OneHotEncoder(sparse = False)
c_X_train = ohe.fit_transform(c_X_train)

ohe = OneHotEncoder(sparse = False)
c_X_test = ohe.fit_transform(c_X_test)


# In[430]:


# Normalize numerical data on train and test data, respectively
n_X_train = ss_X.fit_transform(n_X_train)
n_X_test = ss_X.fit_transform(n_X_test)

y_train = ss_X.fit_transform(y_train.reshape(-1,1))
y_test = ss_X.fit_transform(y_test.reshape(-1,1))


# In[431]:


X_train = np.hstack((c_X_train, n_X_train))
X_test = np.hstack((c_X_test, n_X_test))


# # 3 Determine the model type

# ## 3.1 Try Linear regression 

# In[438]:


# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)

# Prediction
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

fs = pd.DataFrame({"columns": list(['holiday', 'Sun','Mon','Tue','Wed','Thu','Fri','Sat', 'weekday', 'weekend','Sun and cloudy', 'frog and cloudy ', 'rain and snow','strong rain and heavy snow','temp', 'atemp',
       'hum', 'windspeed']), "coef":list((lr.coef_.T))})
fs.sort_values(by=['coef'],ascending=False)


# ## 3.1.1 Model valiadtion
# 
# 

# In[439]:


# Applying r2_score to validate the performence of linearregression
print( 'The r2 score of LinearRegression on test is', r2_score(y_test, y_test_pred_lr))
print('The r2 score of LinearRegression on train is', r2_score(y_train, y_train_pred_lr))


# In[445]:


# residuals distribution on the traing set
f, ax = plt.subplots(figsize=(7,5))
f.tight_layout()
ax.hist(y_train - y_train_pred_lr, bins=40, label = 'Residual Linear', color = 'b', alpha = .5);
ax.set_title("Histogram of Residuals")
ax.legend(loc = 'best');


# In[446]:


plt.figure(figsize=(4, 3))
plt.scatter(y_train, y_train_pred_lr)
plt.plot([-3, 3], [-3, 3], '--k') # 3 sigma is fine cause of the normalization
plt.axis('tight')
plt.xlabel('True number')
plt.ylabel('Predicted number')
plt.tight_layout()


# # 3.2 Ridge Regression

# In[448]:


from sklearn.linear_model import RidgeCV

# set haperparameter
alphas = [ 0.01, 0.1, 1, 10, 100]

#train
ridge = RidgeCV(alphas = alphas, store_cv_values= True)
ridge.fit(X_train, y_train)

#predicted
y_test_pred_ridge = ridge.predict(X_test)
y_train_pred_ridge = ridge.predict(X_train)

#validation
print( 'The r2 score of RidgeCV on test is', r2_score(y_test, y_test_pred_ridge))
print('The r2 score of RidgeCV on train is', r2_score(y_train, y_train_pred_ridge))


# In[463]:


mse_mean = np.mean(ridge.cv_values_, axis = 0)
plt.plot(np.log10(alphas), mse_mean.reshape(len(alphas),1))

#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30]) # for optimal parameters
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print('alpha is:', ridge.alpha_)

fs = pd.DataFrame({"columns": list(['holiday', 'Sun','Mon','Tue','Wed','Thu','Fri','Sat', 'weekday', 'weekend','Sun and cloudy', 'frog and cloudy ', 'rain and snow','strong rain and heavy snow','temp', 'atemp',
       'hum', 'windspeed']), "coef_lr":list((lr.coef_.T)), "coef_ridge":list((ridge.coef_.T))})
fs.sort_values(by=['coef_lr'],ascending=False) 


# ### The comparasion of the linear regression and ridge regression clearly indictes that witout the regularization the ourfitted value would make the coefficients very large, however the r2_score is about being the same. This is a issue needed to be considered.

# # 3.3 Lasso

# In[466]:


from sklearn.linear_model import LassoCV

# set haperparameter
#alphas = [ 0.01, 0.1, 1, 10, 100]

#train
lasso = LassoCV()
lasso.fit(X_train, y_train)

#predicted
y_test_pred_lasso = lasso.predict(X_test)
y_train_pred_lasso = lasso.predict(X_train)

#validation
print('The r2 score of LassoCV on test is', r2_score(y_test, y_test_pred_lasso))
print('The r2 score of LASSOCV on train is', r2_score(y_train, y_train_pred_lasso))


# In[467]:


mses = np.mean(lasso.mse_path_, axis = 1)
plt.plot(np.log10(lasso.alphas_), mses)

#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30]) # for optimal parameters
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print('alpha is:', lasso.alpha_)

