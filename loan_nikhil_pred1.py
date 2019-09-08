import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

training_data = pd.read_csv("/home/nikhil/Analytic_vidya/train.csv")
testing_data = pd.read_csv("/home/nikhil/Analytic_vidya/test.csv")
testing_id = testing_data.Loan_ID
trainig_data1 = training_data.drop("Loan_Status",axis=1)
target = training_data.Loan_Status
dataset = pd.concat([trainig_data1,testing_data],axis=0)
dataset = dataset.drop("Loan_ID",axis=1)
target = target.map({"Y":1,"N":0})
#===fill na value=====#
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')

string = ["Gender","Married","Education","Self_Employed"]
for i in range(4):
    dataset[string[i]] = dataset[string[i]].fillna(max(dataset[string[i]]))

number = ["LoanAmount","Loan_Amount_Term","Credit_History"]
for i in range(3):
    dataset[number[i]] = dataset[number[i]].fillna(dataset[number[i]].dropna().median())


dataset["Dependents"] = dataset["Dependents"].map({"0":0,"1":1,"2":2,"3+":3})
dataset["Dependents"] = dataset["Dependents"].fillna(dataset["Dependents"].dropna().median())

from sklearn.preprocessing import LabelBinarizer 
encoder = LabelBinarizer()
label_binerizer = encoder.fit_transform(dataset["Dependents"].values.reshape(-1,1))
label_binerizer = pd.DataFrame(label_binerizer[:,1:])
all_data = dataset.drop("Dependents",axis=1)
data = pd.concat([all_data.reset_index(),label_binerizer],axis=1)

data.Gender = data.Gender.map({"Male":0,"Female":1})
data.Married = data.Married.map({"Yes":0,"No":1})
data.Education = data.Education.map({"Graduate":1,"Not Graduate":0})
data.Self_Employed = data.Self_Employed.map({"No":0,"Yes":1})
data.Property_Area = data.Property_Area.map({"Semiurban":1,"Urban":2,"Rural":0})

INCOME = 0.9*data["ApplicantIncome"] + 0.2*data["CoapplicantIncome"]
data["INCOME"] = INCOME
R = 0.01
EMI = (data["LoanAmount"]*0.01*(1.01)**data["Loan_Amount_Term"])/(1.01**(data["Loan_Amount_Term"]))
data["EMI"] = EMI

train_data = data.iloc[range(614),:]
test_data = data.iloc[np.arange(614,data.shape[0]),:]

#==================training model====================#
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
rdm_clf = RandomForestClassifier(n_estimators=200,random_state=42)
ext_clf = ExtraTreesClassifier(n_estimators=200,random_state=42)
svm_clf = SVC(probability=True,random_state=42)
mlp_clf = MLPClassifier(random_state=42)
gbd_clf = LogisticRegression(random_state=42)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_data,target,test_size=0.3,stratify=target,random_state=42)

from imblearn.over_sampling import ADASYN
sm = ADASYN(random_state=42)
X_res,y_res = sm.fit_sample(X_train, y_train)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
dst_clf = DecisionTreeClassifier()
bag_clf = BaggingClassifier(dst_clf,n_estimators=300,max_samples=100,bootstrap=True,n_jobs=-1)
bag_clf.fit(X_res,y_res)
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)



rdm_clf = RandomForestClassifier(n_estimators=500,random_state=42)
rdm_clf.fit(X_train,y_train)
from sklearn.metrics import f1_score
error = [f1_score(y_test,y_pred) for y_pred in rdm_clf.staged_predict(X_test)]   #it use to find best n_estimatores velue
best_n_estimators = np.argmin(error)





from sklearn.ensemble import VotingClassifier
named_estimators = [("rdm_clf",rdm_clf),("ext_clf",ext_clf),("svm_clf",svm_clf),("mlp_clf",mlp_clf),("gbd_clf",gbd_clf)]
voting_clf = VotingClassifier(named_estimators,voting="soft")
voting_clf.fit(X_train,y_train)
acc_score = [estimator.score(X_test,y_test) for estimator in voting_clf.estimators_]

voting_clf.score(X_test, y_test)









