# --------------
#Importing header files

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split


# Code starts here
data=pd.read_csv(path)
X=data.drop(['customer.id','paid.back.loan'],axis=1)
y=data['paid.back.loan']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
print(fully_paid)
plt.bar(fully_paid.index,height=fully_paid)
plt.show()
# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate']= [float(x[0:-1])/100 for x in X_train['int.rate']]
X_test['int.rate']= [float(x[0:-1])/100 for x in X_test['int.rate']]
num_df = X_train.select_dtypes(include='number')
cat_df = X_train.select_dtypes(include='object')
print(X_test.info())
print(cat_df.head())
# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols = num_df.columns
plt.figure(figsize=(30,20))
fig, axes = plt.subplots(nrows = 9,ncols=1)
for i in range(0,len(cols)):
    sns.boxplot(x=y_train,y=num_df[cols[i]],ax=axes[i])
plt.tight_layout()
plt.show()   
# Code ends here


# --------------
# Code starts here
cols=cat_df.columns
fig,axes = plt.subplots(nrows = 2 , ncols = 2,figsize=(10,20))

for i in range(0,2):
    for j in range(0,2):
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train, ax=axes[i,j])
        fig.tight_layout()
plt.show()

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
cols=cat_df.columns
le=LabelEncoder()
for i in cols:
    le.fit(X_train[i])
    X_train[i]=le.transform(X_train[i])
    X_test[i]=le.transform(X_test[i])

y_train = list(map(lambda x: 1 if x=='Yes' else 0, y_train))
y_test = list(map(lambda x: 1 if x=='Yes' else 0, y_test))

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print(acc)
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2 = DecisionTreeClassifier(random_state=0)
p_tree=GridSearchCV(estimator=model_2, param_grid=parameter_grid, cv=5)
p_tree.fit(X_train,y_train)
acc_2=p_tree.score(X_test,y_test)
print(acc_2)
# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True, class_names=['loan_paid_back_yes','loan_paid_back_no'])
graph_big=pydotplus.graph_from_dot_data(dot_data)
# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


