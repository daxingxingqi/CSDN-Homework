
# coding: utf-8

# # 1. Import Necessary Tool Packages

# In[1077]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

import matplotlib.pyplot as plt
import seaborn as sns
# color = sns.color_palette()

get_ipython().run_line_magic('mathplotlib', 'inline # generate figure inside console')


# # 2. Data explore

# ## 2.1 Import the data

# In[1078]:


# path to where the data lies
dpath =  '/home/zichen/Desktop/CSDN/week1/Ames_House/'
train = pd.read_csv(dpath + "Ames_House_train.csv")
test = pd.read_csv(dpath + "Ames_House_test.csv")
train.head()


# ## 2.2 Basic data information

# ### 2.2.1 Train Data 

# In[1079]:


# Train data information
train.info()
train.shape


# In[1080]:


### Check is there are any nulls
train.isnull().sum().sort_values(ascending = False)


# In[1081]:


## statistic characteritics
train.describe()


# ### 2.2.3 Test Data 

# In[1082]:


# Test data information
test.info()


# In[1083]:


### Check is there are any nulls
test.isnull().sum().sort_values(ascending = False)


# In[1084]:


## statistic characteritics
test.describe()


# ## 2.3 Data washing

# ### 2.3.1 Handle Missing Data with train set

# In[1085]:


### Fill Nan with median
n = np.median(train["LotFrontage"].dropna(how = 'any'))
train["LotFrontage"].fillna(value = n, inplace = True)
# Fill the missing data with the most occurant data
train["GarageFinish"].fillna(value = 'Unf', inplace = True)
train["GarageQual"].fillna(value = 'TA', inplace = True)    
train["GarageCond"].fillna(value = 'TA', inplace = True)  
train["BsmtQual"].fillna(value = 'TA', inplace = True)
train["GarageType"].fillna(value = 'Attchd', inplace = True)


# In[1086]:


train = train.drop(['Id','Alley', 'FireplaceQu', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature','LowQualFinSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch','MiscVal'  ], axis = 1)
train.dropna(how = 'any', inplace = True)


# In[1087]:


train.shape


# ### 2.3.2 Handle Missing Data with test set

# In[1088]:


### Fill Nan with median
t_n = np.median(test["LotFrontage"].dropna(how = 'any'))
train["LotFrontage"].fillna(value = t_n, inplace = True)
# Fill the missing data with the most occurant data
test["GarageFinish"].fillna(value = 'Unf', inplace = True)
test["GarageQual"].fillna(value = 'TA', inplace = True)    
test["GarageCond"].fillna(value = 'TA', inplace = True)  
test["BsmtQual"].fillna(value = 'TA', inplace = True)
test["GarageType"].fillna(value = 'Attchd', inplace = True)
test = test.drop(['Id','Alley', 'FireplaceQu', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature','LowQualFinSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch','MiscVal'], axis = 1)
test.dropna(how = 'any', inplace = True)


# In[1089]:


test.shape


# ## 2.4 Numerical data anlysis

# In[1090]:


# SalePrice histogram/distribution
fig = plt.figure()
sns.distplot(train.SalePrice.values, bins = 30, kde = True)
plt.xlabel('Sale Price', fontsize = 12)
plt.show()
plt.scatter(range(train.shape[0]),train["SalePrice"].values,color='purple')
plt.title("Distribution of Sale Price")


# In[1091]:


# Delet train data > 400000
train = train[train.SalePrice < 500000]


# In[1092]:


# 房屋到街道的直线距离(英尺) histogram/distribution
fig = plt.figure()
sns.distplot(train.LotFrontage.values, bins = 40, kde = True)
plt.xlabel('LotFrontage', fontsize = 12)
plt.show()
plt.scatter(range(train.shape[0]),train["LotFrontage"].values,color='purple')
plt.title("Distribution of LotFrontage")


# In[1093]:


# Delet train data > 130
train = train[train.LotFrontage < 130]


# In[1094]:


# 土地的大小(平方英尺) histogram/distribution
fig = plt.figure()
sns.distplot(train.LotArea.values, bins = 40, kde = True)
plt.xlabel('LotArea', fontsize = 12)
plt.show()
plt.scatter(range(train.shape[0]),train["LotArea"].values,color='purple')
plt.title("Distribution of LotArea")


# In[1095]:


# Delet train data > 20000
train = train[train.LotArea < 20000]


# In[1096]:


# 游泳池面积 histogram/distribution
fig = plt.figure()
sns.distplot(train.PoolArea.values, bins = 40, kde = True)
plt.xlabel('PoolArea ', fontsize = 12)
plt.show()
plt.scatter(range(train.shape[0]),train["PoolArea"].values,color='purple')
plt.title("Distribution of PoolArea ")


# In[1097]:


# Delet train data > 100
train = train[train.PoolArea < 100]


# # 3 Data preparing

# ## 3.1 Label Data Encoding

# In[1098]:


def handle_non_numerical_data(df):
    columns = df.columns.values # get data from each column
    
    for column in columns:
        text_digit_vals = {}   # creat a dictinary to store the categorical data and its number
        def convert_to_int(val):    
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64: # check if the value is a number
            column_contents = df[column].values.tolist()                    # transfer to list
            unique_elements = set(column_contents)                          # set the column to be a collection with no duplicate elements
            x = 0
            for unique in unique_elements:                                  # count each value 
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                    
            df[column] = list(map(convert_to_int, df[column]))
    
    return df


# In[1099]:


# Transfer the non numerical data in both train and test set
handle_train = handle_non_numerical_data(train)
handle_test = handle_non_numerical_data(test)


# In[1100]:


# Split the data into input and output
y_train = handle_train['SalePrice'].values
X_train = handle_train.drop(['SalePrice'], axis = 1 )


# ## 3.2 Data processing

# In[1101]:


# Data split for onehotencoder (categorical and numerical data)
c_X_train =  X_train[['MSSubClass','MSZoning','Street','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle','OverallQual', 'OverallCond','YearBuilt','YearRemodAdd', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond','Foundation','BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical','BsmtFullBath','BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive','MoSold', 'YrSold','SaleType','SaleCondition']].values
n_X_train =  X_train.drop(['MSSubClass','MSZoning','Street','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle','OverallQual', 'OverallCond','YearBuilt','YearRemodAdd', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond','Foundation','BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical','BsmtFullBath','BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive','MoSold', 'YrSold','SaleType','SaleCondition'], axis = 1)

c_X_test =  X_train[['MSSubClass','MSZoning','Street','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle','OverallQual', 'OverallCond','YearBuilt','YearRemodAdd', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond','Foundation','BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical','BsmtFullBath','BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive','MoSold', 'YrSold','SaleType','SaleCondition']].values
n_X_test =  X_train.drop(['MSSubClass','MSZoning','Street','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle','OverallQual', 'OverallCond','YearBuilt','YearRemodAdd', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond','Foundation','BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical','BsmtFullBath','BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive','MoSold', 'YrSold','SaleType','SaleCondition'], axis = 1)


# In[1102]:


# Import normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
ss_X =  StandardScaler()
ss_y =  StandardScaler()
# Onehotencoder

ohe = OneHotEncoder(sparse = False)
c_X_train = ohe.fit_transform(c_X_train)

ohe = OneHotEncoder(sparse = False)
c_X_test = ohe.fit_transform(c_X_test)


# In[1104]:


# Normalize numerical data on train and test data, respectively
n_X_train = ss_X.fit_transform(n_X_train)
n_X_test = ss_X.fit_transform(n_X_test)

y_train = ss_X.fit_transform(y_train.reshape(-1,1))
#y_test = ss_X.fit_transform(y_test.reshape(-1,1))


# In[1105]:


X_train = np.hstack((c_X_train, n_X_train))
X_test = np.hstack((c_X_test, n_X_test))


# # 4 Determine the model type

# ## 4.1 Try Linear regression

# In[1110]:


# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)

# Prediction
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)


# In[1111]:


# Applying r2_score to validate the performence of linearregression
print('The r2 score of LinearRegression on train is', r2_score(y_train, y_train_pred_lr))


# In[1113]:


# residuals distribution on the traing set
f, ax = plt.subplots(figsize=(7,5))
f.tight_layout()
ax.hist(y_train - y_train_pred_lr, bins=40, label = 'Residual Linear', color = 'b', alpha = .5);
ax.set_title("Histogram of Residuals")
ax.legend(loc = 'best');


# In[1114]:


plt.figure(figsize=(4, 3))
plt.scatter(y_train, y_train_pred_lr)
plt.plot([-3, 3], [-3, 3], '--k') # 3 sigma is fine cause of the normalization
plt.axis('tight')
plt.xlabel('True number')
plt.ylabel('Predicted number')
plt.tight_layout()


# ## 4.2 Ridge Regression

# In[1116]:


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
print('The r2 score of RidgeCV on train is', r2_score(y_train, y_train_pred_ridge))


# In[1118]:


mse_mean = np.mean(ridge.cv_values_, axis = 0)
plt.plot(np.log10(alphas), mse_mean.reshape(len(alphas),1))

#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30]) # for optimal parameters
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print('alpha is:', ridge.alpha_)


# ## 3.3 Lasso

# In[1119]:


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
print('The r2 score of LASSOCV on train is', r2_score(y_train, y_train_pred_lasso))


# In[1121]:


mses = np.mean(lasso.mse_path_, axis = 1)
plt.plot(np.log10(lasso.alphas_), mses)

#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30]) # for optimal parameters
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print('alpha is:', lasso.alpha_)

