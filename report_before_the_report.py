#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from env import host, user, password
import acquire
import prepare
import exp_mod
import report_wdate


# # inital data retrieval and analysis

# In[2]:


telco_df=acquire.get_telco_data()


# In[3]:


telco_df.info()


# In[4]:


telco_df.describe(include='object').T


# In[5]:


telco_df.churn


# In[6]:


def initial_data(data):
    telco_df=acquire.get_telco_data()
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


initial_data(telco_df)


# # Prepare Data

# In[8]:


prep_telco=prepare.prep_telco(telco_df)
prep_telco


# In[9]:


prep_telco=prep_telco.drop(columns=['phone_service.1','multiple_lines.1','internet_service_type_id.1','payment_type_id.1','online_backup.1','device_protection.1','tech_support.1','streaming_tv.1','streaming_movies.1','contract_type_id.1','paperless_billing.1','total_charges.1','monthly_charges.1','total_charges.1'],inplace=True)


# In[10]:


telco_train,telco_validate,telco_test=prepare.split_telco(prep_telco)
telco_train


# In[ ]:





# # Explore Data

# In[11]:


telco_train.info()


# In[12]:



telco_train['month']=(telco_train.total_charges/telco_train.monthly_charges)
telco_validate['month']=(telco_validate.total_charges/telco_validate.monthly_charges)
telco_test['month']=(telco_test.total_charges/telco_test.monthly_charges)


# In[13]:


telco_train.month.value_counts()


# In[14]:


telco_train.month.value_counts().nunique()


# In[15]:


telco_train.columns


# In[16]:


telco_train.contract_type


# In[17]:


telco_train[telco_train.month==1].describe().T


# In[18]:


telco_train[telco_train.month>1].describe().T


# In[19]:


telco_train[telco_train.month>1].describe().T==telco_train[telco_train.month==1].describe().T


# In[20]:


telco_train[telco_train.month==1].churn_Yes.value_counts()


# In[21]:


telco_train[telco_train.churn_Yes==1].describe().T


# In[22]:


len(telco_train[telco_train.month==1])/len(telco_train[telco_train.churn_Yes==1])


# In[23]:


telco_train[telco_train.churn_Yes==0].describe().T==telco_train[telco_train.churn_Yes==1].describe().T


# In[24]:


g=sns.JointGrid(data=telco_train, x="month", y="tenure", space=0, ratio=17,hue='churn_Yes')
g.plot_joint(sns.scatterplot, size=telco_train["churn_Yes"], sizes=(50, 120),
             color="g", alpha=.6, legend=False)
g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)


# In[25]:


g=sns.JointGrid(data=telco_train, x="month", y="partner_Yes", space=0, ratio=17,hue='churn_Yes')
g.plot_joint(sns.scatterplot, size=telco_train["churn_Yes"], sizes=(50, 120),
             color="g", alpha=.6, legend=True)
g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)


# In[26]:


telco_train_m=telco_train[telco_train.month<=1]
telco_train_m[telco_train_m.churn_Yes==1].describe().T


# In[27]:


telco_train_m[telco_train_m.churn_Yes==1].contract_type.value_counts()


# In[28]:


222/688


# In[29]:


report_wdate.signup_date_train(telco_train)
report_wdate.signup_date_val(telco_validate)
report_wdate.signup_date_test(telco_test)


# In[30]:


report_wdate.compareid(telco_train).nunique()


# In[31]:


sns.displot(
    data=telco_train,
    x="monthly_charges", hue="churn_Yes",
    kind="kde", height=6,
    multiple="fill", clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)


# In[32]:


sns.violinplot(data=telco_train, x='online_security_Yes',y='churn_Yes',palette="light:g", inner="points", orient="h")


# In[33]:


sns.displot(
    data=telco_train,
    x="tenure", hue="churn_Yes",
    kind="kde", height=6,
    multiple="fill", clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)


# In[34]:


sns.displot(
    data=telco_train,
    x="month", hue="churn_Yes",
    kind="kde", height=6,
    multiple="fill", clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)


# In[35]:


sns.violinplot(data=telco_train, x='signup_month',y='churn_Yes',palette="light:g", inner="points", orient="h")


# In[36]:


sns.lineplot(data=telco_train, x="signup_month", y="month",hue='churn_Yes')


# In[37]:


sns.relplot(x="signup_month", y="contract_type", hue="churn_Yes",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=telco_train)


# In[38]:


sns.relplot(x="month", y="contract_type", hue="churn_Yes",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=telco_train)


# In[39]:


stats.mannwhitneyu(telco_train.month, telco_train.tenure)


# # Model

# In[40]:


telco_train.churn_Yes.mode()


# In[41]:


telco_x_train = telco_train.select_dtypes(exclude=['object']).drop(columns=['churn_Yes'])
telco_y_train = telco_train.select_dtypes(exclude=['object']).churn_Yes

telco_x_validate = telco_validate.select_dtypes(exclude=['object']).drop(columns=['churn_Yes'])
telco_y_validate = telco_validate.select_dtypes(exclude=['object']).churn_Yes

telco_x_test = telco_test.select_dtypes(exclude=['object']).drop(columns=['churn_Yes'])
telco_y_test = telco_test.select_dtypes(exclude=['object']).churn_Yes


# In[42]:


(telco_y_train==0).mean()


# In[43]:


clf_telco = DecisionTreeClassifier(max_depth=3, random_state=123)
clf_telco = clf_telco.fit(telco_x_train, telco_y_train)


# In[44]:


plt.figure(figsize=(13, 7))
plot_tree(clf_telco, feature_names=telco_x_train.columns, rounded=True)


# In[45]:


telco_y_pred = pd.DataFrame({'churn': telco_y_train,'baseline': 0, 'model_1':clf_telco.predict(telco_x_train)})
telco_y_pred


# In[46]:


y_pred_proba = clf_telco.predict_proba(telco_x_train)
y_pred_proba[0:5]


# In[47]:


print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf_telco.score(telco_x_train, telco_y_train)))


# In[48]:


confusion_matrix(telco_y_pred.churn, telco_y_pred.model_1)


# In[49]:


print(classification_report(telco_y_pred.churn,telco_y_pred.model_1))


# In[50]:


pd.DataFrame(confusion_matrix(telco_y_pred.churn, telco_y_pred.model_1), index=['actual_notchurn','acutal_churn'], columns=['prep_notchurn','prep_churn'])


# In[51]:


telco_TN = 2923
telco_FP = 207
telco_FN = 642
telco_TP = 453


# In[52]:


telco_all = telco_TP + telco_FP + telco_FN + telco_TN
telco_acc = (telco_TP + telco_TN) / telco_all

telco_TurePositiveRate = telco_recall = telco_TP/ (telco_TP + telco_FN)

telco_FalsePositiveRate = telco_FP / (telco_FP + telco_TN)

telco_TrueNegativeRate = telco_TN / (telco_TN + telco_FP)

telco_FalseNegativeRate = telco_FN / (telco_FN + telco_TP)

telco_precision = telco_TP / (telco_TP + telco_FP)

telco_f1_score = 2 * (telco_precision*telco_recall) / (telco_precision+telco_recall)

telco_support_pos = telco_TP + telco_FN
telco_support_neg = telco_FP + telco_TN


# In[53]:


print('accuracy is:',telco_acc,'Ture Positive Rate is:',telco_TurePositiveRate,'False Positive Rate is:',telco_FalsePositiveRate,'/n',
      'True Negative Rate is:',telco_TrueNegativeRate,'False Negative Rate is:',telco_FalseNegativeRate,'precision is:',telco_precision,'/n',
      'f1_score is:',telco_f1_score,'support_pos is:',telco_support_pos,'support_neg is:',telco_support_neg)


# In[54]:


print(classification_report(telco_y_train, telco_y_pred.model_1))


# In[55]:


# random forest


# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=10, 
                            random_state=123)
rf.fit(telco_x_train, telco_y_train)


# In[57]:


clf_telco.score(telco_x_train,telco_y_train)


# In[58]:


telco_y_predict=rf.predict(telco_x_train)


# In[59]:


print(classification_report(telco_y_train,telco_y_predict))


# In[60]:


confusion_matrix(telco_y_train,telco_y_predict)


# In[61]:


ConfusionMatrixDisplay(confusion_matrix(telco_y_train,telco_y_predict),display_labels=rf.classes_).plot()


# In[62]:


TN = 3018
FP = 112
FN = 293
TP = 802


# In[63]:


all = TP + FP + FN + TN
acc = (TP + TN) / all

TurePositiveRate = recall = TP/ (TP + FN)

FalsePositiveRate = FP / (FP + TN)

TrueNegativeRate = TN / (TN + FP)

FalseNegativeRate = FN / (FN + TP)

precision = TP / (TP + FP)

f1_score = 2 * (precision*recall) / (precision+recall)

support_pos = TP + FN
support_neg = FP + TN


# In[64]:


print('accuracy is:',acc,'Ture Positive Rate is:',TurePositiveRate,'False Positive Rate is:',FalsePositiveRate,'/n',
      'True Negative Rate is:',TrueNegativeRate,'False Negative Rate is:',FalseNegativeRate,'precision is:',precision,'/n',
      'f1_score is:',f1_score,'support_pos is:',support_pos,'support_neg is:',support_neg)


# In[65]:


for model in range (2,11):
    rf=RandomForestClassifier(max_depth=model, random_state=123)
    rf=rf.fit(telco_x_train,telco_y_train)
    y_predict=rf.predict(telco_x_train)
    print('model depth',model)
    print(classification_report(telco_y_train, telco_y_predict))


# In[66]:


model=[]
for num in range (2,20):
    rf=RandomForestClassifier(max_depth=num,random_state=123)
    rf=rf.fit(telco_x_train,telco_y_train)
    train_accuracy=rf.score(telco_x_train,telco_y_train)
    validate_accuracy=rf.score(telco_x_validate,telco_y_validate)
    result = {
        "max_depth": num,
        "train_accuracy": train_accuracy,
         "validate_accuracy": validate_accuracy
    }
    model.append(result)
test_validate = pd.DataFrame(model)
test_validate["difference"] = test_validate.train_accuracy - test_validate.validate_accuracy
test_validate


# In[67]:


sns.relplot(x='max_depth',y='difference',data=test_validate)


# In[68]:


model=[]
max_depth=25
for num in range (2,max_depth):
    mdepth=max_depth-num
    min_leaf=num
    rf=RandomForestClassifier(max_depth=mdepth,min_samples_leaf=min_leaf,random_state=123)
    rf=rf.fit(telco_x_train,telco_y_train)
    train_accuracy=rf.score(telco_x_train,telco_y_train)
    validate_accuracy=rf.score(telco_x_validate,telco_y_validate)
    result = {
        'min_samples_leaf':min_leaf,
        'max_depth': mdepth,
        "train_accuracy": train_accuracy,
        "validate_accuracy": validate_accuracy
    }
    model.append(result)
test_validate = pd.DataFrame(model)
test_validate["difference"] = test_validate.train_accuracy - test_validate.validate_accuracy
test_validate


# In[69]:


sns.relplot(x='max_depth',y='difference',data=test_validate)


# In[70]:


knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(telco_x_train, telco_y_train)


# In[71]:


y_pred= knn.predict(telco_x_train)
y_valid=knn.predict(telco_x_validate)


# In[72]:


print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(telco_x_train, telco_y_train)))


# In[73]:


model=[]
for num in range (2,20):
    knn = KNeighborsClassifier(n_neighbors=num, weights='uniform')
    knn=knn.fit(telco_x_train, telco_y_train)
    train_accuracy=knn.score(telco_x_train,telco_y_train)
    validate_accuracy=knn.score(telco_x_validate,telco_y_validate)
    result = {
        "max_depth": num,
        "train_accuracy": train_accuracy,
        "validate_accuracy": validate_accuracy
    }
    model.append(result)
test_validate = pd.DataFrame(model)
test_validate["difference"] = test_validate.train_accuracy - test_validate.validate_accuracy
test_validate


# In[74]:


sns.relplot(x='max_depth',y='difference',data=test_validate)


# In[75]:




# In[ ]:




