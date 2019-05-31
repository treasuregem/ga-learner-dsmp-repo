# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Code starts here
df=pd.read_csv(path)
#print(df.head(5))
#print(df.info)
cols=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)
print(df.head(5))
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
count=y.value_counts()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)
# Code ends here


# --------------
# Code starts here
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
X_train[cols] = X_train[cols].apply(pd.to_numeric, errors='coerce')
X_test[cols] = X_test[cols].apply(pd.to_numeric, errors='coerce')
print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(X_train.dtypes)
# Code ends here


# --------------
# Code starts here

X_train.dropna(subset = ['YOJ','OCCUPATION'],inplace=True)

X_test.dropna(subset = ['YOJ','OCCUPATION'],inplace=True)
#y_train=y_train.drop(index=X_train.index)
#y_test=y_test.drop(index=X_test.index)
y_train=y_train[X_train.index]
y_test=y_test[X_test.index]
cols=['AGE','CAR_AGE','INCOME','HOME_VAL']
for i in cols:
    X_train[cols]=X_train[cols].fillna(X_train[cols].mean())
    X_test[cols]=X_test[cols].fillna(X_test[cols].mean())

print(y_train.shape)
print(y_test.shape)

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]
# Code starts here
le=LabelEncoder()
for i in range(0,len(columns)):
    X_train[columns[i]]=le.fit_transform(X_train[columns[i]])
    X_test[columns[i]]=le.transform(X_test[columns[i]])
print(X_train.head())
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# code starts here 
model=LogisticRegression(random_state = 6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
print(score)
print(precision)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=9)
X_train,y_train=smote.fit_sample(X_train,y_train)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
# Code ends here


