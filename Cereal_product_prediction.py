#!/usr/bin/env python
# coding: utf-8

# Task 
# cereal rating prediction 
# given data about different breakfast cereals , let's try to predict the rating of given cereal
# 
# We will use a linear regression model to make our predictions

# # Import

# In[59]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso


# In[60]:


data=pd.read_csv('cereal.csv')
data.head(20)


# In[61]:


data.info()


# # Preprocessing

# In[62]:


data= data.drop('name',axis=1)


# MISSING VALUES

# In[63]:


(data == -1).sum()


# In[64]:


data=data.replace(-1,np.NaN)


# In[65]:


data.isna().sum()


# In[66]:


for column in ['carbo','sugars','potass']:
    data[column]=data[column].fillna(data[column].mean())


# In[67]:


data.isna().sum()


# # Encoding

# In[68]:


{column : list(data[column].unique()) for column in ['mfr','type']}


# In[69]:


data['type']=data['type'].apply(lambda x: 1 if x=='H' else 0)


# In[70]:


#One-Hot Encoder the 'mfr' column

dummies = pd.get_dummies(data['mfr'])
dummies


# In[71]:


#concat
data=pd.concat([data,dummies],axis=1)
data=data.drop('mfr',axis=1)


# In[72]:


data


# # Splitting and Scaling

# In[73]:


y=data.loc[:,'rating']
X=data.drop('rating',axis=1)


# In[74]:


X


# In[75]:


scaler=StandardScaler()
X=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)


# In[76]:


X


# In[78]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.70,random_state=42)


# # Training

# In[79]:


model=LinearRegression()
l2_model=Ridge(alpha=1.0)
l1_model=Lasso(alpha=1.0)


# In[82]:


model.fit(X_train,y_train)
l2_model.fit(X_train,y_train)
l1_model.fit(X_train,y_train)

print("Models Trained.")


# In[83]:


model_r2=model.score(X_test,y_test)
l2_model_r2=l2_model.score(X_test,y_test)
l1_model_r2=l1_model.score(X_test,y_test)


# In[85]:


print('R^2 Score\n' + "*" * 10)
print("    Without Regularization :",model_r2)
print("with L2 (Ridge) Regularization", l2_model_r2)
print("with L1 (lasso) Regularization", l1_model_r2)


# In[ ]:




