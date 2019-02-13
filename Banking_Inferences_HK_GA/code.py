# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  

# path        [File location variable]
data=pd.read_csv(path)
#Code starts here
data_sample=data.sample(n=sample_size,random_state=0)
sample_mean=data_sample['installment'].mean()
sample_std=data_sample['installment'].std()
margin_of_error=z_critical*sample_std/math.sqrt(sample_size)
confidence_interval=(sample_mean-margin_of_error,sample_mean+margin_of_error)
print(confidence_interval)
true_mean=data['installment'].mean()
print(true_mean)











# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])
number_of_smaples=1000
#Code starts here
fig,axes=plt.subplots(nrows=3,ncols=1)

for i in range(len(sample_size)):
    m=[]
    for j in range(number_of_smaples):
        sample=data.sample(n=sample_size[i])
        m.append(sample['installment'].mean())
    mean_series=pd.Series(m)
    print(mean_series)
    axes[i].plot(mean_series)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
#print(data['int.rate'].mean())
data['int.rate'].replace('%',"",regex=True,inplace=True)
data['int.rate']=data['int.rate'].astype(float)
data['int.rate']=data['int.rate']/100
#print(data['int.rate'].mean())
#print(data['int.rate'])
z_statistic,p_value=ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
print(f"z_statistic: {z_statistic}")
print(f"p_value: {p_value}")
if(p_value>0.05):
    inference='Accept'
else:
    inference='Reject'

print(inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value=ztest(data[data['paid.back.loan']=='No']['installment'],data[data['paid.back.loan']=='Yes']['installment'])
print(z_statistic)
print(p_value)
if(p_value>0.05):
    inference='Accept'
else:
    inference='Reject'
print(inference)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed=pd.concat([yes,no],keys= ['Yes','No'],axis=1)
#print(observed)
chi2, p, dof, ex=stats.chi2_contingency(observed)
print(chi2)
print(p)
if(chi2 > critical_value):
    inference='Reject'
else:
    inference='Accept'
print(inference)


