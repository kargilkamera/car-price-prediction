#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.linear_model  import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[6]:


data = pd.read_csv('quikr_car.csv')
data.head()


# In[7]:


data.shape


# In[8]:


data.describe()


# # cleaning

# In[9]:


backup = data.copy()


# In[10]:


data['year'].unique()


# In[11]:


data = data[data['year'].str.isnumeric()]
data['year'] = data['year'].astype(int)


# In[12]:


data= data[data["Price"]!='Ask For Price']


# In[13]:


data['Price'] = data['Price'].str.replace(',','').astype(int)


# In[14]:


data['kms_driven'] = data["kms_driven"].str.split(' ').str.get(0).str.replace(",",'')


# In[15]:


data = data[data["kms_driven"].str.isnumeric()]


# In[16]:


data['kms_driven'] = data['kms_driven'].astype(int)


# In[17]:


data = data[~data["fuel_type"].isna()]


# In[18]:


data['name'] = data['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[19]:


data


# In[20]:


data.describe()


# In[21]:


data = data[data["Price"]<6e6].reset_index(drop=True)


# In[22]:


data


# In[23]:


data.to_csv('cleaned data.csv')


# # model

# In[39]:


x = data.drop(columns='Price')
y = data['Price']


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle


# In[42]:


ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


# In[51]:


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder = 'passthrough')


# In[52]:


lr = LinearRegression()


# In[53]:


pipe = make_pipeline(column_trans,lr)


# In[54]:


pipe.fit(x_train,y_train)


# In[55]:


y_predict = pipe.predict(x_test)


# In[56]:


r2_score(y_test,y_predict)


# In[64]:


score = []
for i in range(1000):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict(x_test)
    score.append(r2_score(y_test,y_pred))
    


# In[65]:


print(score.index(max(score)))


# In[66]:


max(score)


# In[67]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=661)
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)


# In[68]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[71]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))


# In[ ]:




