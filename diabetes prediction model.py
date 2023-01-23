#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[5]:


data = pd.read_csv('diabetes.csv')
data
# for data visualization purposes. 


# In[9]:


#To view the visualization of the data

sns.pairplot(data= data, hue = 'Outcome')


# In[11]:


#heatmap and correlation

sns.heatmap(data.corr(), annot = True)
plt.show()


# In[23]:


#in order to filter out zero values, they are replaced by nulls

data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
data["Glucose"].fillna(data["Glucose"].mean(), inplace = True)
data["BloodPressure"].fillna(data["BloodPressure"].mean(), inplace = True)
data["SkinThickness"].fillna(data["SkinThickness"].mean(), inplace = True)
data["Insulin"].fillna(data["Insulin"].mean(), inplace = True)
data["BMI"].fillna(data["BMI"].mean(), inplace = True)

#Feature scaling in order to use the specific algorithm
sc = MinMaxScaler(feature_range = (0, 1))
data_scaled = sc.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled)
X = data_scaled.iloc[:, [1, 4, 5, 7]].values
Y = data_scaled.iloc[:, 8].values
# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = data['Outcome'] )


# In[24]:


# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)
# Making predictions on test dataset
Y_pred = svc.predict(X_test)

#Evaluation
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: " + str(accuracy * 100))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm
Output:
array([[87, 13],
       [20, 34]], dtype=int64)
# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)

