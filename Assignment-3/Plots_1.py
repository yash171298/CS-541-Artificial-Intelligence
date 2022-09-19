#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from itertools import cycle, islice


# In[15]:


n=["n=1","n=2","n=3","n=4","n=5","n=6","n=7"]
ted= [313.90,90.65,82.12,121.44,208.19,358.33,594.25]
red= [549.54,221.05,277.55,491.95,913.89,1629.46,2733.56]
news=[707.12,279.18,358.45,665.89,1318.88,2533.91,4652.04]


# In[16]:


df= pd.DataFrame(ted)
df["N-GRAMS"]= n
df["test_ted"]= ted
df["test_reddit"]= red
df["test_news"]= news
df.drop([0],axis=1)


# In[17]:


my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))
df.plot(x="N-GRAMS",y=["test_ted","test_reddit","test_news"],kind="bar",rot=0, color=my_colors)


# In[5]:


n_=["n=1","n=2","n=3","n=4","n=5","n=6","n=7"]


# In[21]:


ted2=[270.98,114.17,145.48,259.98,482.64,853.75,1426.02]
red2=[347.54,138.22,174.89,309.25,565.72,1007.35,1673.27]
news2=[425.15,234.74,358.30,711.47,1409.25,2713.39,4977.61]


# In[22]:


d_f= pd.DataFrame(test__ted)
d_f["N-GRAMS"]= n_
d_f["test_ted"]= ted2
d_f["test_reddit"]= red2
d_f["test_news"]= news2
d_f.drop([0],axis=1)


# In[23]:


d_f.plot(x="N-GRAMS",y=["test_ted","test_reddit","test_news"],kind="bar",rot=0, color=my_colors)


# In[ ]:




