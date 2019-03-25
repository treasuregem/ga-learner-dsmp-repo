#-------------------------------------------
	import pandas as pd
	from sklearn import preprocessing
	
	#path : File path
	
	# Code starts here
	
	# read the dataset
	dataset=pd.read_csv(path)
	print(dataset.head())
	
	# look at the first five columns
	print(dataset.iloc[:,0:5])
	print(dataset.columns)
	# Check if there's any column which is not useful and remove it like the column id
	dataset.drop('Id',axis=1,inplace=True)
	print(dataset.columns)
	# check the statistical description
	print(dataset.info())
	print(dataset.describe())
#---------------------------------------------
	# We will visualize all the attributes using Violin Plot - a combination of box and density plots
	import seaborn as sns
	from matplotlib import pyplot as plt
	
	#names of all the attributes 
	cols=dataset.columns
	
	#number of attributes (exclude target)
	size=len(cols)-1
	
	#x-axis has target attribute to distinguish between classes
	x = cols[size]
	
	#y-axis shows values of an attribute
	y = cols[0:size]
	
	#Plot violin for all attributes
	for i in range(size):
	    sns.violinplot(y=y[i],x=x,data=dataset)
	    plt.show()
#-------------------------------------------------
	import numpy
	threshold = 0.5

	# no. of features considered after ignoring categorical variables

	num_features = 10

	# create a subset of dataframe with only 'num_features'
	subset_train=dataset.iloc[:,0:num_features]
	cols=subset_train.columns
	#print(cols)
	#print(subset_train.head())
	#Calculate the pearson co-efficient for all possible combinations
	data_corr = subset_train.corr()
	#print(data_corr)
	sns.heatmap(data_corr,annot=True)
	plt.show()
	# Set the threshold and search for pairs which are having correlation level above threshold
	corr_var_list=data_corr[(data_corr.abs() > 0.5) & (data_corr.abs() < 1)]
	corr_var_list.dropna(how='all',inplace=True)
	#print(corr_var_list)
	sns.heatmap(corr_var_list,annot=True)
	plt.show()
	temp_list=[]
	# Sort the list showing higher ones first 
	for i in range(0,corr_var_list.shape[0]):
		for j in range(0,corr_var_list.shape[0]):
			if(i >= j):
				if not pd.isnull(corr_var_list.iloc[i][j]):
					temp_list.append([corr_var_list.iloc[i][j],cols[i],cols[j]])
	#print(temp_list)
	#print('-----------------------------------------------')
	corr_var_list=temp_list
	s_corr_list=sorted(corr_var_list,key=lambda lst: -abs(lst[0]))
	#Print correlations and column names
	print("Correlations: ")
	for corr,i,j in s_corr_list:
	   print ("%s and %s = %.2f" % (j,i,corr))
#-----------------------------------------------------------------
	#Import libraries 
	from sklearn import cross_validation
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split
	# Identify the unnecessary columns and remove it 
	dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
	X=dataset.drop('Cover_Type',axis=1)
	Y=dataset['Cover_Type']
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.2)
	
	# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
	
	#Standardized
	#Apply transform only for non-categorical data
	scale=StandardScaler()
	X_train_temp = scale.fit_transform(X_train.iloc[:,:10])
	X_test_temp = scale.fit_transform(X_test.iloc[:,:10])
	
	#Concatenate non-categorical data and categorical
	X_train1=numpy.concatenate([X_train_temp,X_train.iloc[:,10:]],axis=1)
	X_test1=numpy.concatenate([X_test_temp,X_test.iloc[:,10:]],axis=1)
	scaled_features_train_df = pd.DataFrame(X_train1, index=X_train.index, columns=X_train.columns)
	scaled_features_test_df = pd.DataFrame(X_test1, index=X_test.index, columns=X_test.columns)
#----------------------------------------------------------------
	from sklearn.feature_selection import SelectPercentile
	from sklearn.feature_selection import f_classif
	
	# Write your solution here:
	skb=SelectPercentile(score_func=f_classif,percentile=20)
	predictors=skb.fit_transform(X_train1,Y_train)
	scores=predictors.tolist()
	#print(scores)
	top_k_index=skb.get_support(indices=True)
	print(top_k_index)
	top_k_predictors=predictors[top_k_index]
	print(top_k_predictors)
#---------------------------------------------------------------
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
	
	clf=OneVsRestClassifier(LogisticRegression())
	clf1=OneVsRestClassifier(LogisticRegression())
	
	model_fit_all_features = clf1.fit(X_train,Y_train)
	predictions_all_features = clf1.predict(X_test)
	
	score_all_features = accuracy_score(Y_test,predictions_all_features)
	print(score_all_features)
	#print(top_k_predictors)
	#print(scaled_features_train_df.columns[skb.get_support()])
	#print(scaled_features_train_df.loc[:,skb.get_support()])
	X_train_top_k = scaled_features_train_df.loc[:,skb.get_support()]
	X_test_top_k = scaled_features_test_df.loc[:,skb.get_support()]
	
	model_fit_top_features = clf.fit(X_train_top_k,Y_train)
	predictions_top_features= clf.predict(X_test_top_k)
	
	score_top_features = accuracy_score(Y_test,predictions_top_features)
	print(score_top_features)
#---------------------------------------------------------