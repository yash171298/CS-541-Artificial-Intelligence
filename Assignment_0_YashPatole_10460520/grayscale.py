#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.image import imread

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

try:
    img = mpimg.imread('Documents/black-puppy-adorable.jpg')     
    gray = rgb2gray(img)    
    imgplot = plt.imshow(gray, cmap = plt.get_cmap(name = 'gray'))
    plt.show()
#plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#plt.show()
except FileNotFoundError:
    print('File does not exist')


# In[ ]:




