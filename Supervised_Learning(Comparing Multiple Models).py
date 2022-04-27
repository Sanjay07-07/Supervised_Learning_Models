#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading Required Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Loading the Bank Dataset(csv file)

bank = pd.read_csv("D:\Work\Imarticus\Machine Learning\Exam 1\\bank.csv", sep=';')


# In[3]:


#Preview the dataset

bank.head()


# In[4]:


bank.shape #We see there are 41,188 observations and 21 columns


# In[5]:


bank.info()
#We see most of the columns are of object type and few numerical data types.


# In[6]:


#We are building a pre-defined function to show how categorical values are disributed and plotting them in bar-plot.
def bar_plot(variable):
    # temp df 
    temp = pd.DataFrame()
    # count categorical values
    temp['No_deposit'] = bank[bank['y'] == 'no'][variable].value_counts()
    temp['Yes_deposit'] = bank[bank['y'] == 'yes'][variable].value_counts()
    temp.plot(kind='bar')
    plt.xlabel(f'{variable}')
    plt.ylabel('Number of clients')
    plt.title('Distribution of {} and deposit'.format(variable))
    plt.show();


# In[7]:


#We are plotting few important variables
bar_plot('job'), bar_plot('marital'), bar_plot('education'), bar_plot('contact'), bar_plot('loan'), bar_plot('housing')


# Primary analysis of several categorical features reveals:
# 
# Administrative staff and technical specialists opened the deposit most of all. In relative terms, a high proportion of pensioners and students might be mentioned as well.
# 
# Although in absolute terms married consumers more often agreed to the service, in relative terms the single was responded better.
# 
# Best communication channel is cellular.
# 
# The difference is evident between consumers who already use the services of banks and received a loan.
# Home ownership does not greatly affect marketing company performance.

# In[8]:


bank1 = bank.copy()


# In[9]:


#Exploratory Data Analysis

#Using Pre-Defined function to treat the Null Values.

def null_values(base_dataset):
    print(base_dataset.isna().sum())
    ##null value percentage
    null_value_table = (base_dataset.isna().sum()/base_dataset.shape[0])*100
    ## null value percentage beyond thershold drop, else treat the columns
    retained_columns = null_value_table[null_value_table<0].index
    # if any variable as null value greater than input(like 30% of the data) value than those variable
    #are considered as drop
    drop_columns = null_value_table[null_value_table>30].index
    base_dataset.drop(drop_columns, axis = 1, inplace = True)
    len(base_dataset.isna().sum().index)
    cont = base_dataset.describe().columns
    cat = [i for i in base_dataset.columns if i not in base_dataset.describe().columns]
    for i in cat:
        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace = True)
    for i in cont:
        base_dataset[i].fillna(base_dataset[i].median(),inplace = True)
    print(base_dataset.isna().sum())
    return base_dataset, cat, cont


# In[10]:


bank1, cat, cont = null_values(bank)


# In[11]:


#Convert target variable into numeric
bank1.y = bank1.y.map({'no':0, 'yes':1}).astype('int64')


# In[12]:


#Dummy variable Declaration

dummy_columns = []
for i in bank1.columns:
    if (bank1[i].nunique()>=3) & (bank1[i].nunique()<5):
        dummy_columns.append(i)
        
#Dummy Variable
dummies_tables = pd.get_dummies(bank1[dummy_columns])

#Adding the dummy variables to our dataset.
for i in dummies_tables.columns:
    bank1[i] = dummies_tables[i]
    
#Displaying columns after dummy variable creation
bank1.columns


# In[13]:


#Drop the existing columns after the ceation of dummy variable for those
bank1 = bank1.drop(dummy_columns, axis=1)


# In[14]:


#Checking if the columns are dropped.
bank1.columns


# In[15]:


#Label Encoder
from sklearn.preprocessing import LabelEncoder
def label_encoders(data, cat):
    le=LabelEncoder()
    for i in cat :
        le.fit(data[i])
        x = le.transform(data[i])
        data[i] = x
    return data


# In[16]:


bank_new = bank1
cat = bank1.describe(include="object").columns

label_encoders(bank_new, cat).head()


# In[17]:


#Correlation between variables
plt.figure(figsize=(20,10))
sns.heatmap(bank_new.corr(), annot= True)


# In[18]:


#FEATURE SELECTION

# Split the data into 40% test set and 60% training set

from sklearn.model_selection import train_test_split
x = bank_new.drop("y", axis=1)
y = bank_new["y"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4, random_state=0)


# In[19]:


#Using RandomClassifier for feature selection
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000,random_state=0, n_jobs=-1)

clf.fit(x_train, y_train)


# In[20]:


feat_labels = x.columns.values

#Pre-defined function to print the name and importance of each feature
feature_importance = []
for feature in zip(feat_labels, clf.feature_importances_):
    feature_importance.append(feature)


# In[21]:


feature_importance


# In[22]:


from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(clf, threshold = 0.01)


# In[23]:


sfm.fit(x_train, y_train)


# In[24]:


#Print the name of the most important features
selected_features = []

for feature_list_index in sfm.get_support(indices=True):
    selected_features.append(feat_labels[feature_list_index])
    
selected_features


# In[25]:


data_selected = bank_new[selected_features]
data_selected.head()


# In[26]:


#Handling unbalanced data using SMOTE

from collections import Counter
#summarize class distribution
counter = Counter(y)
print(counter)

#Data is imbalanced.


# In[27]:


from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
#transform the dataset
oversample = SMOTE()
x, y = oversample.fit_resample(x,y)
counter = Counter(y)
print(counter)

#Data is now balanced.


# In[28]:


#Standardizing the data using the one of the scalers provided by sklearn

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_selected)


# In[29]:


data_standardized = scaler.fit_transform(data_selected)


# We are build the following Supervised Learning models:
# a. Logistic Regression
# b. AdaBoost
# c. Naïve Bayes
# d. KNN
# e. SVM
# 
# and then tabulating the metrics to compare which model is the best.

# In[30]:


#Splitting the data
x = data_standardized
y = bank_new["y"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=10)

# print the shape of 'x_train'
print('x_train:', x_train.shape)

# print the shape of 'x_test'
print('x_test:', x_test.shape)

# print the shape of 'y_train'
print('y_train:', y_train.shape)

# print the shape of 'y_test'
print('y_test:', y_test.shape)


# In[31]:


#AdaBoost
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(random_state=10)
adaboost.fit(x_train,y_train)


# In[32]:


y_pred_adaboost = adaboost.predict(x_test)


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_adaboost))


# In[34]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

plt.rcParams['figure.figsize']=(8,5)

fpr,tpr,th = roc_curve(y_test,y_pred_adaboost)

plt.plot(fpr,tpr)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])

plt.plot([0,1],[0,1],'r--') 

plt.text(x=0.05,y=0.8,s=('AUC Score',round(roc_auc_score(y_test,y_pred_adaboost),4)))

plt.xlabel('False positive rate(1-Specificity)')
plt.ylabel('True positive rate(Specificity)')

plt.grid(True)


# In[35]:


from sklearn import metrics

cols = ['Model','Precision Score','Recall Score','Accuracy Score','f1-score']

result_tabulation = pd.DataFrame(columns=cols)

adaboost = pd.Series({"Model":"AdaBoost",
                        'Precision Score': metrics.precision_score(y_test,y_pred_adaboost,average="macro"),
                        'Recall Score': metrics.recall_score(y_test,y_pred_adaboost,average="macro"),
                        'Accuracy Score': metrics.accuracy_score(y_test,y_pred_adaboost),
                        'f1-score': metrics.f1_score(y_test,y_pred_adaboost,average="macro")})

result_tabulation = result_tabulation.append(adaboost , ignore_index=True)
result_tabulation


# In[36]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
knn.fit(x_train, y_train)


# In[37]:


y_pred_knn = knn.predict(x_test)


# In[38]:


print(classification_report(y_test,y_pred_knn))


# In[39]:


knn_mod = pd.Series({"Model":"KNN",
                        'Precision Score': metrics.precision_score(y_test,y_pred_knn,average="macro"),
                        'Recall Score': metrics.recall_score(y_test,y_pred_knn,average="macro"),
                        'Accuracy Score': metrics.accuracy_score(y_test,y_pred_knn),
                        'f1-score': metrics.f1_score(y_test,y_pred_knn,average="macro")})

result_tabulation = result_tabulation.append(knn_mod , ignore_index=True)
result_tabulation


# In[40]:


#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 1)
classifier.fit(x_train,y_train)


# In[41]:


y_pred_svm = classifier.predict(x_test)


# In[42]:


print(classification_report(y_test,y_pred_svm))


# In[43]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_svm)
print("confusion_matrix\n",cm)
print('Lenght of t_test :',len(y_test))
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\n Accuracy of SVM for the given Dataset : ",accuracy)


# In[44]:


svm_mod = pd.Series({"Model":"SVM",
                        'Precision Score': metrics.precision_score(y_test,y_pred_svm,average="macro"),
                        'Recall Score': metrics.recall_score(y_test,y_pred_svm,average="macro"),
                        'Accuracy Score': metrics.accuracy_score(y_test,y_pred_svm),
                        'f1-score': metrics.f1_score(y_test,y_pred_svm,average="macro")})

result_tabulation = result_tabulation.append(svm_mod , ignore_index=True)
result_tabulation


# In[45]:


#NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)


# In[46]:


y_pred_nb = gnb.predict(x_test)
y_pred_nb


# In[47]:


print(classification_report(y_test,y_pred_nb))


# In[48]:


nb_mod = pd.Series({"Model":"Naive Bayes",
                        'Precision Score': metrics.precision_score(y_test,y_pred_nb,average="macro"),
                        'Recall Score': metrics.recall_score(y_test,y_pred_nb,average="macro"),
                        'Accuracy Score': metrics.accuracy_score(y_test,y_pred_nb),
                        'f1-score': metrics.f1_score(y_test,y_pred_nb,average="macro")})

result_tabulation = result_tabulation.append(nb_mod , ignore_index=True)
result_tabulation


# In[49]:


#Logistic Regreassion
from sklearn.linear_model import LogisticRegression


# In[50]:


model = LogisticRegression()


# In[51]:


model.fit(x_train,y_train)


# In[52]:


y_pred_log = model.predict(x_test)


# In[53]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_log))


# In[54]:


log_reg = pd.Series({"Model":"Logistic-Regression",
                        'Precision Score': metrics.precision_score(y_test,y_pred_log,average="macro"),
                        'Recall Score': metrics.recall_score(y_test,y_pred_log,average="macro"),
                        'Accuracy Score': metrics.accuracy_score(y_test,y_pred_log),
                        'f1-score': metrics.f1_score(y_test,y_pred_log,average="macro")})

result_tabulation = result_tabulation.append(log_reg , ignore_index=True)
result_tabulation


# *Here we can see that from our 5 models the best model is Logistic Reg. Although the accuracy of all the 5 models are more or less near to other models. All our models have high and good accuracy of more than 85%. We can use any of these models to predict further and will more or less a similar results but in the market as we choose the best we will either go with AdaBoost or Logistic Regression models.

# In[ ]:




