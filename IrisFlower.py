#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\AASHIMA\\Desktop\\Python\\irisflowe.csv")


# In[2]:


data.head()


# In[3]:


data.describe()


# In[4]:


data.species.unique()


# In[5]:


data["Target"]=data.species.map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
df0=data[0:50]
df1=data[50:100]
df2=data[100:]


# In[6]:


plt.xlabel("Sepal Lenght")
plt.ylabel("Sepal Width")
plt.scatter(df0["sepal_length"],df0['sepal_width'],color='green', marker='+')
plt.scatter(df1["sepal_length"],df1['sepal_width'],color='blue', marker='.')


# In[7]:


plt.xlabel("Petal Lenght")
plt.ylabel("Petal  Width")
plt.scatter(df0["petal_length"],df0['petal_width'],color='green', marker='+')
plt.scatter(df1["petal_length"],df1['petal_width'],color='blue', marker='.')


# In[8]:


import plotly.express as px
fig=px.scatter(data,x="sepal_width",y="sepal_length",color="species")
fig.show()


# In[9]:


fig=px.scatter(data,x="petal_width",y="petal_length",color="species")
fig.show()


#  # It can be concluded that Iris-Setsos', sepal_width and lenght are small as compared to other two. Iris-Virginica's both sepal lenght and width are largest among all the spieces

# In[10]:


x = data.drop("species", axis=1)
y = data["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
print(knn.score(x_test,y_test))


# In[11]:


len(x_train)


# In[12]:


len(x_test)


# In[14]:


x_new = np.array([[20.0, 15.3, 11.5, 1.2,2.0]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[16]:


x_new = np.array([[5, 2.9, 1, 0.2,0.5]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[17]:


from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
cm


# In[18]:


import seaborn as sns
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel("Prerdicted")
plt.ylabel("Truth")
#Diagonal values are correct prediicted values whereas other values shows that they are incorrect values


# In[19]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




