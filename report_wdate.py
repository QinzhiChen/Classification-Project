#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from env import host, user, password
import acquire
import prepare
import explore


# # Acquire Data and Prepare Data

# In[2]:


telco_df=acquire.get_telco_data_wdate()


# In[3]:


telco_df.info()


# In[4]:


telco_df.describe(include='object').T


# In[5]:


telco_df.churn


# In[6]:


def initial_data_wdate(data):
    telco_df=acquire.get_telco_data_wdate()
    print('this data frame has',telco_df.shape[0],'rows and', telco_df.shape[1],'columns')
    print('                        ')
    print(telco_df.info())
    print('                        ')
    print(telco_df.describe())
    print('                        ')
    print(telco_df.describe(include='object').T)
    print('                        ')
    print(telco_df.columns)
    print('ended of initial report')
    print('                        ')


# In[7]:


initial_data_wdate(telco_df)


# In[8]:


telco_train,telco_validate,telco_test=prepare.split_telco(telco_df)
telco_train


# In[9]:


telco_train.drop(columns=['internet_service_type_id.1','payment_type_id.1','contract_type_id.1'],inplace=True)
telco_validate.drop(columns=['internet_service_type_id.1','payment_type_id.1','contract_type_id.1'],inplace=True)
telco_test.drop(columns=['internet_service_type_id.1','payment_type_id.1','contract_type_id.1'],inplace=True)


# # Explore Data

# In[18]:


telco_train['month']=(telco_train.total_charges/telco_train.monthly_charges)
telco_validate['month']=(telco_train.total_charges/telco_train.monthly_charges)
telco_test['month']=(telco_train.total_charges/telco_train.monthly_charges)


# In[19]:


ndate=telco_train[telco_train.signup_date>='2021-09-21 18:07:34']


# In[12]:


ndate[ndate.month==1].signup_date


# In[13]:


369/1869


# In[14]:


telco_train['signup_month']=pd.DatetimeIndex(telco_train['signup_date']).month


# In[15]:


telco_train['signup_month'].value_counts()


# In[16]:


def signup_date(df):
    df['signup_month']=pd.DatetimeIndex(telco_train['signup_date']).month
    return df


# In[17]:


def compareid(df):
    return df.customer_id==telco_train.customer_id


# In[ ]:





# In[ ]:




