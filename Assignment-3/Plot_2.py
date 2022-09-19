#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from itertools import cycle, islice


# In[30]:


n=["n=1","n=2","n=3","n=4","n=5","n=6","n=7"]
ted= [313.91,91.56,75.53,115.86,209.19,357.33,594.25]
red= [549.54,211.62,241.39,493.95,913.89,1635.46,2743.56]
news=[708.12,247.89,358.11,665.89,1314.88,2512.91,4635.04]


# In[31]:


df= pd.DataFrame(ted)
df["N-GRAMS"]= n
df["test_ted"]= ted
df["test_reddit"]= red
df["test_news"]= news
df.drop([0],axis=1)


# In[32]:


my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))
df.plot(x="N-GRAMS",y=["test_ted","test_reddit","test_news"],kind="bar",rot=3,color=my_colors)


# In[33]:


n_=["n=1","n=2","n=3","n=4","n=5","n=6","n=7"]


# In[35]:


ted2=[270.98,114.17,129.68,269.98,485.64,851.75,1424.02]
red2=[345.54,126.47,174.89,319.25,566.72,1001.35,1670.27]
news2=[428.15,227.81,352.30,702.47,1401.25,2713.39,4839.61]


# In[36]:


d_f= pd.DataFrame(test__ted)
d_f["N-GRAMS"]= n_
d_f["test_ted"]= ted2
d_f["test_reddit"]= red2
d_f["test_news"]= news2
d_f.drop([0],axis=1)


# In[37]:


d_f.plot(x="N-GRAMS",y=["test_ted","test_reddit","test_news"],kind="bar",rot=0, color=my_colors)


# In[ ]:




