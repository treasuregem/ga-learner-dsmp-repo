# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
df_len=len(df)
fico_g700_len=len(df[df['fico']>700])
p_a=fico_g700_len/df_len
print(p_a)
p_b=len(df[df['purpose']=='debt_consolidation'])/df_len
print(p_b)
df1=df[df['purpose']=='debt_consolidation']
df2=df[df['fico']>700]
#print(df1)
p_a_b=len(df1[df1['fico']>700])/len(df1)
p_b_a=len(df2[df2['purpose']=='debt_consolidation'])/len(df2)
print(p_a_b)
print(p_b_a)
result=(p_b_a==p_a)
print(result)

# code ends here


# --------------
# code starts here
prob_lp=len(df[df['paid.back.loan']=='Yes'])/len(df)
print(prob_lp)
prob_cs=len(df[df['credit.policy']=='Yes'])/len(df)
print(prob_cs)
new_df=df[df['paid.back.loan']=='Yes']
prob_pd_cs=len(new_df[new_df['credit.policy']=='Yes'])/len(new_df)
print(prob_pd_cs)
bayes=(prob_pd_cs*prob_lp)/prob_cs
print(bayes)


# code ends here


# --------------
# code starts here
df1=df[df['paid.back.loan']=='No']
print(df1)

# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
plt.hist(df['installment'],bins=20)
plt.axvline(inst_median,color='blue',label='Median:{:.2f}'.format(inst_median))
plt.axvline(inst_mean,color='green',label='Mean:{:.2f}'.format(inst_mean))
plt.legend()
plt.title("Installment Distribution")
plt.xlabel("Installment")
plt.ylabel("Frequency")
plt.show()
plt.hist(df['log.annual.inc'],bins=20)
plt.xlabel("Log Annual Income")
plt.ylabel("Frequency")
plt.title("Annual Income Distribution")
plt.show()
# code ends here


