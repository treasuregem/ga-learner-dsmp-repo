# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
# Load Offers
offers=pd.read_excel(path,sheet_name=0)
transactions=pd.read_excel(path,sheet_name=1)
transactions['n']=1
df=pd.merge(offers,transactions)
print(df.head())
# Load Transactions


# Merge dataframes


# Look at the first 5 rows



# --------------
# Code starts here

# create pivot table
matrix=pd.pivot_table(df,index='Customer Last Name',columns='Offer #',values='n')
matrix.fillna(0,inplace=True)
matrix.reset_index(inplace=True)
print(matrix.head())
# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans


# Code starts here
cluster=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])
print(matrix.head())
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2,random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
print(matrix.head())
# dataframe to visualize clusters by customer names
clusters=pd.DataFrame(data=matrix,columns=['Customer Last Name','cluster','x','y'])
print(clusters.head())
plt.scatter(data=clusters,x='x',y='y',c='cluster',cmap='viridis')
plt.show()
# visualize clusters


# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data=pd.merge(clusters,transactions)
data=pd.merge(offers,data)
champagne={}
for i in data['cluster'].unique():
    new_df=data[data['cluster']==i]
    counts=new_df['Varietal'].value_counts(ascending=False)
    if counts.index[0]=='Champagne':
        champagne[i]=counts[0]
print(champagne)
v_list=list(champagne.values())
k_list=list(champagne.keys())
cluster_champagne=k_list[v_list.index(max(v_list))]
print(cluster_champagne)
# merge `data` and `offers`

# initialzie empty dictionary


# iterate over every cluster

    # observation falls in that cluster

    # sort cluster according to type of 'Varietal'

    # check if 'Champagne' is ordered mostly

        # add it to 'champagne'


# get cluster with maximum orders of 'Champagne' 


# print out cluster number




# --------------
# Code starts here

# empty dictionary
discount={}
for i in data['cluster'].unique():
    new_df=data[data['cluster']==i]
    counts=sum(new_df['Discount (%)'])/new_df.shape[0]
    discount[i]=counts
print(discount)
cluster_discount =max(discount,key=discount.get)
print(cluster_discount)
# iterate over cluster numbers

    # dataframe for every cluster

    # average discount for cluster

    # adding cluster number as key and average discount as value 


# cluster with maximum average discount


# Code ends here


