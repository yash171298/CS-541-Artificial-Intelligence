#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('Assignment 1\income.csv')
print("Shape of the data= ", data.shape)
data.head()


# In[3]:


print(data.isnull().sum())
data['age'] = data['age'].fillna(data['age'].mean())
data['fnlwgt'] = data['fnlwgt'].fillna(data['fnlwgt'].mean())
data['hours-per-week'] = data['hours-per-week'].fillna(data['hours-per-week'].mean())
data['education-num'] = data['education-num'].fillna(data['education-num'].mean())
data['capital-loss'] = data['capital-loss'].fillna(data['capital-loss'].mean())
data['capital-gain'] = data['capital-gain'].fillna(data['capital-gain'].mean())
data.head()


# In[4]:


data.columns


# In[5]:


X_data = data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
X_data = X_data.apply(lambda x : (x - x.min(axis = 0))  / (x.max(axis = 0) - x.min(axis = 0)))
print(X_data.head())


# In[6]:


data['sex'] = pd.factorize(data.sex)[0]
data['race'] = pd.factorize(data.race)[0]
data['workclass'] = pd.factorize(data.workclass)[0]
data['education'] = pd.factorize(data.education)[0]
data['occupation'] = pd.factorize(data.occupation)[0]
data['relationship'] = pd.factorize(data.relationship)[0]
data['native-country'] = pd.factorize(data['native-country'])[0]
data['marital-status'] = pd.factorize(data['marital-status'])[0]
data.head()


# In[7]:


data = data.loc[:, ['age', 'fnlwgt']]
X = data.values
X


# In[8]:


sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('fn')
plt.ylabel('age')
plt.show()


# In[9]:


def calculate_cost(X, centroids, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
    return sum


# In[10]:


def kmeans(X, k):
    diff = 1
    cluster = np.zeros(X.shape[0])
    centroids = data.sample(n=k).values
    while diff:
        for i, row in enumerate(X):
            mn_dist = float('inf')
        # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
            # store closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
     # if centroids are same then leave
    
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster


# In[11]:


cost_list = []
for k in range(1, 10):
    print("hey working")
    centroids, cluster = kmeans(X, k)
    # WCSS (Within cluster sum of square)
    cost = calculate_cost(X, centroids, cluster)
    cost_list.append(cost)
print(cost_list)


# In[12]:


plt.figure(figsize=(5,3))
sns.lineplot(x=range(1,10), y=cost_list, marker='o')
plt.scatter(5,cost_list[4], s = 200, c = 'red', marker='*')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.show()


# In[ ]:





# In[ ]:




