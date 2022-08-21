import os
import env
import pandas as pd
import csv

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_telco_data_wdate():
    filename = "telcowdate.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM customers JOIN customer_signups USING (customer_id)', get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename,index=False)

        # Return the dataframe to the calling code
        return df  

def get_telco_data():
    filename = "telco.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM customers JOIN contract_types USING (contract_type_id) JOIN internet_service_types USING (internet_service_type_id) JOIN customer_subscriptions USING (customer_id) JOIN customer_contracts USING(customer_id) JOIN customer_payments USING (customer_id) JOIN customer_signups USING (customer_id)', get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename,index=False)

        # Return the dataframe to the calling code
        return df  