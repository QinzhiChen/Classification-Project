import os
import env
import pandas as pd
import csv


def get_telco_data():
    filename = "telco.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM customers JOIN customer_subscriptions USING (customer_id) JOIN customer_contracts USING(customer_id) JOIN customer_payments USING (customer_id) JOIN internet_service_types WHERE internet_service_types.internet_service_type_id=customers.internet_service_type_id', get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename,index=False)

        # Return the dataframe to the calling code
        return df  