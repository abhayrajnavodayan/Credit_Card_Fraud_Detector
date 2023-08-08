#!/usr/bin/env python
# coding: utf-8
import pandas as pd 
#Importing pandas library to load the dataset
data=pd.read_csv('creditcard.csv')
data
# ## Showing  all Columns
pd.options.display.max_columns=None
data
# ## 1.Displaying first 5 rows of dataset
data.head()
# ## 2.Displaying last 5 rows of dataset 
data.tail()
# ## 3.Getting shape i.e number of rows and column 
data.shape
print('Number of rows',data.shape[0])
print('Number of columns',data.shape[1])
# ## Getting information about this dataset 
data.info()
# ### i.e in our dataset 30 columns is of float64 and one column is integer type.
# # Data- Preprocessing 
# # 1.Getting null values of dataset 
data.isnull().sum()
# ### since our dataset has no missing value .so we dont have to deal with missing values.
data.head()
# ## 3.Dropping Time column 
data=data.drop('Time',axis=1)
data.head()
# ## 4.Checking for duplicate value 
data.duplicated().any()
# dropping duplicates 
data=data.drop_duplicates()
data.shape
data.head(
)
# ## Analysis on dependent variable 
data['Class'].value_counts()
import seaborn as sns 
sns.countplot(x=data['Class'],data=data)
# ### Note: We can clearly see the dataset is Imbalanced i.e un-evenly distributed.
# # Case 1: Not handling the imbalanced dataset
# ## spliting the dataset into dependent and independent variable 
# 
X=data.drop('Class',axis=1)
y=data['Class']
X
y
# ## Spliting the dataset into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# # Feature Scaling
# we will apply feature scaling for amount column 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train
X_test
X_train['Amount']=sc.fit_transform(pd.DataFrame(X_train['Amount']))
X_test['Amount']=sc.transform(pd.DataFrame(X_test['Amount']))
X_test
X_train
# we will apply feature scaling for amount column 
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount'])) # we use dataframe because fit_transform method expects 2d array as input.
#data.head()
# # Buliding model on different type of classification model 
# ## 1.Logistic Regression model 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred1=classifier.predict(X_test)
# checking accuracy score 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
# ### We can clearly see there is very much difference between accuracy score and recall score,f1 score etc, due to highly imbalanced dataset. and this will true for all model so we have to handle imbalanced dataset.
# # Handling imbalanced dataset
# Undersampling 
# Oversampling
# ## Under Sampling
# ### Undersampling is a technique to balance uneven datasets by keeping all of the data in the minority class and decreasing the size of the majority class
normal=data[data['Class']==0]
fraud=data[data['Class']==1]
normal.shape
fraud.shape
normal_sample=normal.sample(n=473) # normal.sample will select 473 sample randomly.
normal_sample.shape
# ### Now we have 473 normal and 473 fraud transaction.
# ## Concatanating Normal and Fraudulent sampe
new_data=pd.concat([normal_sample,fraud])
new_data['Class'].value_counts()
new_data.head()
# ### We can see the random indexes, now setting the indexes
new_data=pd.concat([normal_sample,fraud],ignore_index=True)
new_data
# # Diving the dataset into dependent and independent variable
X=new_data.drop('Class',axis=1)
y=new_data['Class']
X
# ## Spliting the Dataset into Training and test set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train['Amount']=sc.fit_transform(pd.DataFrame(X_train['Amount']))
X_test['Amount']=sc.transform(pd.DataFrame(X_test['Amount']))
# # Building different models on this balanced dataset 
# ## 1. Logistic Regresson 
# 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred1=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
accuracy_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
# ## 2.Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2=dt.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
accuracy_score(y_test,y_pred2)
recall_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
f1_score(y_test,y_pred2)
# ## 3.Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_estimators=15,criterion='entropy',random_state=0)
rd.fit(X_train,y_train)
y_pred3=rd.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
accuracy_score(y_test,y_pred3)
recall_score(y_test,y_pred3)
precision_score(y_test,y_pred3)
f1_score(y_test,y_pred3)
# ## 4.KNN Algorithm
# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
# knn.fit(X_train,y_train)
# y_pred4=knn.predict(X_test)
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# accuracy_score(y_test,y_pred4)
# precision_score(y_test,y_pred4)
# recall_score(y_test,y_pred4)
# f1_score(y_test,y_pred4)
# ## 5.Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred5=gnb.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy_score(y_test,y_pred5)
precision_score(y_test,y_pred5)
recall_score(y_test,y_pred5)
f1_score(y_test,y_pred5)
# # ### Printing accuracy of all model
# final_data=pd.DataFrame({'Models':['Logistic Regression','DEcision Tree CLassifier','Random Forest Classifier','KNN','Naive Bayes'
#                         ],'Accuracy':[accuracy_score(y_test,y_pred1)*100,
#                                       accuracy_score(y_test,y_pred2)*100,
#                                       accuracy_score(y_test,y_pred3)*100,
#                                     #   accuracy_score(y_test,y_pred4)*100,
#                                       accuracy_score(y_test,y_pred5)*100]})
# final_data
# import seaborn as sns
# sns.barplot(x=final_data['Models'],y=final_data['Accuracy'])
# # ### So We can say that Logistic Regression is best model after under sampling 
# # 2.OVERSAMPLING
import pandas as pd 
dataset=pd.read_csv('creditcard.csv')
dataset
dataset.isnull().sum()
dataset=dataset.drop('Time',axis=1)
dataset.head()
dataset.duplicated().any()
# dropping duplicates 
dataset=dataset.drop_duplicates()
dataset.shape
X=dataset.drop('Class',axis=1)
y=dataset['Class']
X.shape
y.shape
# oversampling by smote 
from imblearn.over_sampling import SMOTE
X_resh,y_resh=SMOTE().fit_resample(X,y)
y_resh.value_counts()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_resh,y_resh,test_size=0.2,random_state=42)
X_train
X_test
# feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train['Amount']=sc.fit_transform(pd.DataFrame(X_train['Amount']))
X_test['Amount']=sc.transform(pd.DataFrame(X_test['Amount']))
X_train
# # Differnt model 
# ## 1.Logistics Regression  
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred1=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
accuracy_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
# ## 2.Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2=dt.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
accuracy_score(y_test,y_pred2)
recall_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
f1_score(y_test,y_pred2)
# ## 3.Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_estimators=15,criterion='entropy',random_state=0)
rd.fit(X_train,y_train)
y_pred3=rd.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
accuracy_score(y_test,y_pred3)
recall_score(y_test,y_pred3)
precision_score(y_test,y_pred3)
f1_score(y_test,y_pred3)
# ## 4.KNN Algorithm 
# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
# knn.fit(X_train,y_train)
# y_pred4=knn.predict(X_test)
# from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
# accuracy_score(y_test,y_pred4)
# recall_score(y_test,y_pred4)
# precision_score(y_test,y_pred4)
# f1_score(y_test,y_pred4)
# ## 5.Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred5=gnb.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
accuracy_score(y_test,y_pred5)
recall_score(y_test,y_pred5)
precision_score(y_test,y_pred5)
f1_score(y_test,y_pred5)
# final_data=pd.DataFrame({'Models':['Logistic Regression','DEcision Tree CLassifier','Random Forest Classifier','KNN','Naive Bayes'
#                         ],'Accuracy':[accuracy_score(y_test,y_pred1)*100,
#                                       accuracy_score(y_test,y_pred2)*100,
#                                       accuracy_score(y_test,y_pred3)*100,
#                                     #   accuracy_score(y_test,y_pred4)*100,
#                                       accuracy_score(y_test,y_pred5)*100]})
# final_data
# sns.barplot(x=final_data['Models'],y=final_data['Accuracy'])
# ### So we can conclude that Random Forest Classifier is best model .
# # Saving the best Model
# ### Traning our best model on entire dataset.
rf1=RandomForestClassifier()
rf1.fit(X_resh,y_resh)
import joblib
import warnings
warnings.filterwarnings("ignore")
joblib.dump(rf1,'credit_card_model')  
# Now our model is successfully saved.so in future we can do prediction by loading model.
model=joblib.load('credit_card_model')
pred=model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1]])
pred
# i.e it is a normal transaction.
if pred==0:
    print('Normal Transaction')
else:
    print('Fraudulent Transaction')
