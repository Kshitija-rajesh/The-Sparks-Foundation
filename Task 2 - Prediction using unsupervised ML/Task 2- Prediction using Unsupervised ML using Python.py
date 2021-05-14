#!/usr/bin/env python
# coding: utf-8

# # TASK2 
# 
# 
# # Prediction using Unsupervised ML
# 
# # From the give 'Iris' dataset, predict the optimum number of clusters and represent it visually.
# 
# Loading the necessary libraries

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


# Loading the dataset

# In[4]:


from sklearn import datasets


# In[33]:


iris= datasets.load_iris()
df = pd.read_csv('E:\R\Task 2- prediction using Unsupervised ML-R\Iris.csv')


# In[34]:


df.head()


# The dataset is successfully loaded.

# Dataset size

# Dataset information

# EDA

# In[36]:


df.shape


# Data description

# In[37]:


df.describe()


# In[38]:


df.info()


# The features sepal legthn and sepal width are slighthly skewed.

# EDA
# 
# Univariate Analysis

# In[39]:


df.corr()


# In[44]:


def outlier_detection(df):
    #Detetcting the Null values and removing them first
    #to emsure that the numerical columns can be detected coreectly.
    r = []
    for col in df.columns:
        for i in df.index:
            if df.loc[i, col]=='NUll' or df.loc[i, col] == np.nan:
                r.append(i)
    df = df.drop(list(set(r)))
    df = df.reset_index()
    df = df.drop('index', axis=1)
    
    #finding out the columns having numerical values.
    num_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except ValueError:
                pass
            
    #Removing the row having values which can be called outliers
    #on the basis of their z-scores of >3 or <-3
    count = 0
    t = []
    for i in num_cols:
        z = np.abs(stats.zscores(df[i]))
        for j in range(len(z)):
            if z[j]>3 or z[j]<-3:
                t.append(j)
                count+=1
    df = df.drop(list(set(t)))
    df = df.reset_index()
    df = df.drop('index', axis=1)
    print(count)
    return df


# In[45]:


df = outlier_detection(df)


# Data Visualization

# In[46]:


sns.countplot(df['Species'])


# In[47]:


sns.catplot("Species", "PetalLengthCm", data = df)


# In[48]:


sns.pairplot(data=df)


# The Elbow graph to find k

# In[54]:


x = df.iloc[:, [1, 2, 3]].values
inertias = []

for i in range(1, 8):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x)
    inertias.append(kmeans.inertia_)
    
plt.plot(range(1, 8), inertias)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[55]:


x


# In[58]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[59]:


#Predict the cluster labels
labels = kmeans.predict(x)


# In[60]:


labels


# In[73]:


plt.scatter(x[labels == 0, 0], x[labels == 0,1],
            s = 100, c='yellow', label = 'Iris setosa')
plt.scatter(x[labels == 1,0], x[labels == 1,1],
            s = 100, c = 'blue', label = 'Iris versicolour')
plt.scatter(x[labels == 2,0], x[labels == 2,1],
            s = 100, c = 'green', label= 'Iris virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'red', label = 'Centriods')

plt.legend()


# In[74]:


Species = ['Iris-setosa', 'Iris-versicolour','Iris-virginica']
Species_ = []
for i in labels:
    Species_.append(Species[i])


# In[75]:


Species_


# In[76]:


df['Predicted_Species'] = Species_


# In[77]:


sns.countplot(df['Predicted_Species'])


# In[78]:


df.head()


# In[ ]:




