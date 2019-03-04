# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)

data = data[data['Rating']<6]
plt.hist(data['Rating'],bins=20)

#plt.hist(data['Rating'],bins=20)
#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
percent_null=(total_null/data.isnull().count())
#print(percent_null)
#print(total_null*100/len(data))
missing_data=pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)
print('-------------------------------------------------------')
#Cleaning
data.dropna(inplace=True)
total_null_1=data.isnull().sum()
percent_null_1=(total_null_1/data.isnull().count())
missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x="Category",y="Rating",data=data, kind="box",height = 10)
plt.title("Rating vs Category [BoxPlot]")
plt.xticks(x=data['Category'], rotation='vertical')
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())
data['Installs'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
data['Installs']=data['Installs'].astype(int)
print(data['Installs'].head())

le=LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
print(data['Installs'].value_counts())

sns.regplot(x='Installs', y='Rating', data=data)
plt.title('Rating vs Installs [RegPlot]')
plt.show()
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype(float)
print(data['Price'].head())
#plt.hist(data['Price'])
sns.regplot(x='Price',y='Rating',data=data)
plt.title('Rating vs Price [RegPlot]')
#Code ends here


# --------------

#Code starts here

print(data['Genres'].unique())
print(data['Genres'].head())
#print(data['Genres'].value_counts())
data['Genres']=data['Genres'].str.split(';').str[0]
print(data['Genres'].head())

gr_mean=(data.groupby('Genres',as_index=False)['Rating'].mean())
print(gr_mean.describe())
gr_mean=(gr_mean.sort_values(by=['Rating']))
print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here
print(data['Last Updated'].head())

data['Last Updated']=pd.to_datetime(data['Last Updated'])
print(data['Last Updated'].head())
max_date=max(data['Last Updated'])
print(max_date)

data['Last Updated Days']= (max_date - data['Last Updated']).dt.days
print(data['Last Updated Days'].head())

sns.regplot(x='Last Updated Days', y='Rating', data=data)
plt.title('Rating vs Last Updated [RegPlot]')
plt.show()
#Code ends here


