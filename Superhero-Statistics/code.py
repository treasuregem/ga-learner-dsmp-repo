# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
#print(data)
data['Gender'].replace('-','Agender',inplace=True)
#print(data[data['Gender']=='Agender'])
gender_count=data['Gender'].value_counts()
print(gender_count)
plt.bar(x=data['Gender'].unique(),height=gender_count)
plt.show()
#Code starts here 




# --------------
#Code starts here
alignment=data['Alignment'].value_counts()
print(alignment)
plt.pie(alignment, labels=['Good','Bad','Neutral'],autopct='%1.1f%%')


# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
ic_df=data[['Intelligence','Combat']]
#print(ic_df)
sc_covariance=sc_df['Strength'].cov(sc_df['Combat'])
ic_covariance=ic_df['Intelligence'].cov(ic_df['Combat'])
#print(sc_covariance)

sc_strength=sc_df['Strength'].std()
sc_combat=sc_df['Combat'].std()
ic_intelligence=ic_df['Intelligence'].std()
ic_combat=ic_df['Combat'].std()

sc_pearson=sc_covariance/(sc_strength*sc_combat)
ic_pearson=ic_covariance/(ic_intelligence*ic_combat)
print(f"Strength and Combat: {sc_pearson}")
#print(sc_df['Strength'].corr(sc_df['Combat'],method='pearson'))

#ic_pearson=data['Intelligence'].corr(data['Combat'])
#print(f"Intelligence and Combat: {ic_pearson}")
print(f"Intelligence and Combat: {ic_pearson}")



# --------------
#Code starts here
total_high=data['Total'].quantile(q=0.99)
#print(total_high)
super_best=data[data['Total']>total_high]
#print(super_best)
super_best_names=super_best['Name'].tolist()
print(super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3)=plt.subplots(3,1,figsize=(10, 10))
fig.tight_layout()

ax_1.boxplot(data['Intelligence'],vert=False)
ax_1.set_title('Intelligence')

ax_2.boxplot(data['Speed'],vert=False)
ax_2.set_title('Speed')

ax_3.boxplot(data['Power'],vert=False)
ax_3.set_title('Power')
plt.show()


