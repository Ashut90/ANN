#!/usr/bin/env python
# coding: utf-8

# In[119]:


from sklearn import datasets


# In[120]:


import numpy as np 


# In[121]:


import pandas as pd
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[122]:


df = datasets.load_boston()


# In[128]:


dataset=pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset.head()


# In[129]:


dataset['Price'] = df.target #add one feature as a dependent feature 


# In[130]:


dataset.head()


# In[134]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)


# In[154]:


#Linear Regression Algo 

from sklearn import linear_model
lin_reg = linear_model.LinearRegression()
from sklearn.model_selection import cross_val_score
mse = cross_val_score(lin_reg, X_train ,y_train,scoring = 'neg_mean_squared_error',cv = 5)
mean_mse = np.mean(mse)
print(mean_mse)
lin_reg.fit(X_train , y_train)


# In[155]:


from sklearn.model_selection import train_test_split


# In[156]:


#Ridge Regression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[157]:


ridge = Ridge()
parms = {'alpha': [1e-12, 1e-10,1e-8, 1e-3,1,3,5,10,12,20]}
ridge_regressor = GridSearchCV(ridge, parms,scoring = 'neg_mean_squared_error', cv= 5)
ridge_regressor.fit(X_train,y_train)


# In[158]:


print(ridge_regressor.best_params_) 
print(ridge_regressor.best_score_)


# In[159]:


from sklearn.model_selection import train_test_split 


# In[160]:


#lasso Regression 

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
parms = {'alpha': [1e-12, 1e-10,1e-8, 1e-3,1,3,5,10,12,20,25,35,50,75,80,100]}
lasso_regressor = GridSearchCV(ridge, parms,scoring = 'neg_mean_squared_error', cv= 15)
lasso_regressor.fit(X_train,y_train)


# In[161]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[162]:


y_pred = lasso_regressor.predict(X_test)
from sklearn.metrics import r2_score

r2_score1 = r2_score(y_pred , y_test)


# In[163]:


print(r2_score1)


# In[164]:


y_pred = ridge_regressor.predict(X_test)
from sklearn.metrics import r2_score

r2_score1 = r2_score(y_pred , y_test)


# In[165]:


print(r2_score1)


# In[166]:


y_pred = lin_reg.predict(X_test)
from sklearn.metrics import r2_score

r2_score1 = r2_score(y_pred , y_test)


# In[167]:


print(r2_score1)


# In[ ]:




