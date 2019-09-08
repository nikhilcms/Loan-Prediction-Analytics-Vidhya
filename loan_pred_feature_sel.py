#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:41:58 2019

@author: nikhil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,auc,confusion_matrix,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb 
leb = LabelEncoder()
train_data=pd.read_csv('/home/nikhil/Analytic_vidya/train.csv')
test_data = pd.read_csv("/home/nikhil/Analytic_vidya/test.csv")

train_data.Loan_Status = train_data.Loan_Status.map({"Y":1,"N":0})
target = train_data.Loan_Status
test_ID = test_data.Loan_ID

train_data = train_data.drop(["Loan_Status"],axis=1)

combined = pd.concat([train_data,test_data],axis=0)
combined = combined.drop("Loan_ID",axis=1)

string = ["Gender","Married","Education","Self_Employed","Dependents","Property_Area"]
for i in range(6):
    combined[string[i]] = combined[string[i]].fillna(max(combined[string[i]]))


number = ["LoanAmount","Loan_Amount_Term","Credit_History"]
for i in range(3):
    combined[number[i]] = combined[number[i]].fillna(combined[number[i]].dropna().median())

for j in range(6):
    combined[string[j]] = leb.fit_transform(combined[string[j]])


INCOME = 0.9*combined["ApplicantIncome"] + 0.2*combined["CoapplicantIncome"]
combined["INCOME"] = INCOME
R = 0.01
EMI = (combined["LoanAmount"]*0.01*(1.01)**combined["Loan_Amount_Term"])/(1.01**(combined["Loan_Amount_Term"]))
combined["EMI"] = EMI
WEIGHT = combined["LoanAmount"]/((combined["ApplicantIncome"]) + combined["CoapplicantIncome"])
combined["WEIGHT"] = WEIGHT



#corr = combined.corr()
#sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)


train = combined.iloc[0:614,:] 
test = combined.iloc[614:981,:] 

clf = ExtraTreesClassifier(n_estimators=500)
clf = clf.fit(train, target)
clf.feature_importances_ 
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(train)
test = model.transform(test)
test1 = pd.DataFrame(test1)
train1 = pd.DataFrame(X_new)
corr = train1.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)

train_df = pd.concat([train1,target],axis=1)
corr = train_df.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)

k = 7
cols = train_df.corr().nlargest(k, 'Loan_Status')['Loan_Status'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.heatmap(cm,vmax=0.8,square=True,cbar=True,yticklabels=cols.values, xticklabels=cols.values)
hm = sns.heatmap(cm, cbar=True,square=True,yticklabels=cols.values, xticklabels=cols.values)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(train, target)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_ros,y_ros,test_size=0.20,random_state=42,stratify = y_ros)

model_GB = GradientBoostingClassifier()
model_LR = LogisticRegression()
model_XGB = XGBClassifier(n_estimatores=500,min_samples_leaf=50)
model_SV = SVC()
model_RF = RandomForestClassifier()

model_XGB.fit(X_train,y_train)
predict = model_XGB.predict(X_test)
predicted_score = accuracy_score(predict,y_test)
scores = cross_val_score(model_XGB,X_train,y_train,cv=5)
auc_score = roc_auc_score(predict,y_test)

'''
gbm_param_grid = {'learning_rate': np.arange(0.05,1.05,.05),'n_estimators': [250,350],'subsample': np.arange(0.05,1.05,.05)}
gbm = XGBClassifier(random_state=12)
randomized_mse = RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid, n_iter=25,scoring='neg_mean_squared_error',cv=4,verbose=1)

randomized_mse.fit(X_train,y_train)
predict = randomized_mse.predict(X_test)
predicted_score = accuracy_score(predict,y_test)
scores = cross_val_score(randomized_mse,X_train,y_train,cv=5)
auc_score = roc_auc_score(predict,y_test)
'''

'''
tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-2,1e-3,1e-4,1e-5],
                     'C':[0.001,0.10,0.1,10,25,50,100,1000]},
                    {'kernel':['sigmoid'],'gamma':[1e-2,1e-3,1e-4,1e-5],
                     'C':[0.001,0.10,0.1,10,25,50,100,1000]},
                    {'kernel':['linear'],'C':[0.001,0.10,0.1,10,25,50,100,1000]}
                   ]
tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7],
                     'C':[0.001,0.10,0.1,1,10,25,50,100,1000]}]

model_SV = GridSearchCV(SVC(C=1),tuned_parameters, cv=5,n_jobs=2)
model_SV.fit(X_train,y_train)
predict = model_SV.predict(X_test)
predicted_score = accuracy_score(predict,y_test)
scores = cross_val_score(model_SV,X_train,y_train,cv=5)
auc_score = roc_auc_score(predict,y_test)
model_SV.fit(X_ros, y_ros)
'''

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [3,5,7],
    'min_samples_leaf': [4,20,50],
    'min_samples_split': [4, 7, 10],
    'n_estimators': [100, 200, 300, 500]
            }
grid_search = GridSearchCV(estimator = model_RF, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)
predict = grid_search.predict(X_test)
predicted_score = accuracy_score(predict,y_test)
scores = cross_val_score(grid_search,X_train,y_train,cv=5)
auc_score = roc_auc_score(predict,y_test)
grid_search.fit(X_ros, y_ros)
model_RF = RandomForestClassifier(bootstrap =True,min_samples_leaf =4,n_estimators =500,min_samples_split =10,max_features=5,max_depth=80)
model_RF.fit(train1,target)
test_predict = model_RF.predict(test)

acc = []
for i in range(len(test_predict)):
	if test_predict[i] == 1:
		acc.append("Y")
	else:
		acc.append("N")
	
test_result = pd.DataFrame(dict(Loan_ID = test_ID,Loan_Status= acc ))
test_result.to_csv("loanPred_online8.csv", encoding='utf-8', index=False)

#{'bootstrap': True, 'min_samples_leaf': 4, 'n_estimators': 300, 'min_samples_split': 10, 'max_features': 7, 'max_depth': 80}
