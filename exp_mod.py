import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



def modeling(telco_x_train, telco_y_train, telco_x_validate, telco_y_validate):
    logrreg_m= LogisticRegression(random_state=123)
    dectre_m = DecisionTreeClassifier(max_depth=3, random_state=123)
    ranfor_m = RandomForestClassifier(max_depth=5, random_state=123)
    knn_m= KNeighborsClassifier(n_neighbors=13, weights='uniform')
    models = [logrreg_m, dectre_m, ranfor_m, knn_m]
    for model in models:
        model.fit(telco_x_train, telco_y_train)
        actual_train = telco_y_train
        pred_train = model.predict(telco_x_train)
        actual_val = telco_y_validate
        pred_val = model.predict(telco_x_validate)
        print(model)
        print('                           ')
        print('train score: ')
        print(classification_report(actual_train, pred_train))
        print('val score: ')
        print(classification_report(actual_val, pred_val))
        print('                        ')
    return logrreg_m, dectre_m, ranfor_m, knn_m