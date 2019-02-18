# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df=pd.read_csv(path)
print(df.head())
y=df.list_price
X=df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']]
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=6)
# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here      
cols=X_train.columns
fig,axes=plt.subplots(nrows=3,ncols=3)

for i in range(3):
    for j in range(3):
        col=cols[i*3+j]
        axes[i,j].scatter(X_train[col],y_train)
        axes[i,j].set_title(col)
        axes[i,j].set_ylabel('list_price')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()



# code ends here



# --------------
from functools import reduce
# Code starts here
corr=(X_train.corr(method ='pearson'))
col=corr.columns
lst=[]
#print(col)
#print(X_test.columns)
num_cols=corr.shape[1]
for i in range(num_cols):
    temp=(corr[(corr[col[i]]>0.75) & (corr[col[i]]!=1)].index.values)
    if(temp.size>0):
        lst.append(temp.tolist())
corr_cols=(list(set(reduce(lambda x,y :x+y ,lst))))
print(corr_cols)
for i in range(len(corr_cols)-1):
    X_train.drop(corr_cols[i],axis=1,inplace=True)
    X_test.drop(corr_cols[i],axis=1,inplace=True)
print(X_train.columns)
print(X_test.columns)
#print(corr)

# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"mse:{mse}")
print(f"r2:{r2}")
print(y_test-y_pred)
X_test.columns
# Code ends here


# --------------
# Code starts here
#residual=(1/len(y_test))*((y_test-y_pred)**2)
residual=(y_test-y_pred)
#print(residual)
plt.hist(residual)
plt.show()

# Code ends here


