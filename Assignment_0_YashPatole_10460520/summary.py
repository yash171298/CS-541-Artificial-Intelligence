#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import csv
import math
sum = 0
max = 0
x = []

data = pd.read_csv("Documents/us-states.csv")
state_name = input("Enter the state for calculating summary:")
data2 = data[data['state'] == state_name]
length = len(data2['cases'])
for i in data2['cases']:
    sum += i
    mean = sum / length
print("The mean of cases in ",state_name, " is ",mean)
for i in data2['cases']:
    if(i>max):
         max = i
print("The maximum cases in ",state_name, " is ",max)
for i in data2['cases']:
    x.append(i)
print("The minimum cases in ",state_name, " is ",min(x))

difference_squared = 0
for i in data2['cases']:
    difference_squared += (i - mean) ** 2
    
std_dev = math.sqrt(difference_squared / ((len(data2['cases'])) - 1))
print("The standard Deviation of cases in ",state_name, " is ",std_dev)


# In[51]:





# In[ ]:





# In[ ]:




