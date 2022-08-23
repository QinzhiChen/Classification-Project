#!/usr/bin/env python
# coding: utf-8

# prep exercise
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import acquire
import warnings
import pandas as pd

# telco data

def prep_telco(df):
    df_telco=acquire.get_telco_data()
    df_telco.drop(columns=['internet_service_type_id','contract_type_id','payment_type_id',
                       ],inplace=True)
    dummy_df_telco = pd.get_dummies(df_telco[['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type','gender']], dummy_na=False, drop_first=[True])
    df_telco=pd.concat([df_telco ,dummy_df_telco],axis=1)
    df_telco.drop(columns=['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type','gender'],inplace=True)
    df_telco.total_charges=df_telco.total_charges.str.replace(' ','0').astype('float')
    return df_telco

def prep_telcowdate(df):
    df_telco=acquire.get_telco_data_wdate()
    df_telco.drop(columns=['internet_service_type_id','contract_type_id','payment_type_id',
                       ],inplace=True)
    dummy_df_telco = pd.get_dummies(df_telco[['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type','gender']], dummy_na=False, drop_first=[True])
    df_telco=pd.concat([df_telco ,dummy_df_telco],axis=1)
    df_telco.drop(columns=['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type','gender'],inplace=True)
    df_telco.total_charges=df_telco.total_charges.str.replace(' ','0').astype(float)
    return df_telco

def split_telco(df):
    df_telco=prep_telco(df)
    train, telco_test = train_test_split(df_telco, test_size=.2, 
                               random_state=123)
    telco_train, telco_validate = train_test_split(train, test_size=.25, 
                 random_state=123)
    
    return telco_train, telco_validate, telco_test





# %%
