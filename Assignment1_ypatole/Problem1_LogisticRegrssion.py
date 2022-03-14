#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = pd.read_csv("Documents/Assignment 1/swad_train.csv")
print(data)


# In[3]:


data.isnull().sum()


# In[4]:


tweets = data['Tweet']
data['Tweet'] = data['Tweet'].str.lower()
data['Tweet'] = data['Tweet'].str.replace( "@user", " ", regex=True) 


# In[5]:




f = open("Documents/Assignment 1/punctuations.txt", "r")
punctuations = f.read()
punctuations = punctuations.split('\n')
for i in punctuations:
  data['Tweet'] = data['Tweet'].str.replace( i, " %s "%(i,), regex=True) 
  data['Tweet'] = data['Tweet'].str.replace( i, "", regex=True) 
print(data)

      


# In[6]:


f  = open("Documents/Assignment 1/stopwords.txt", "r")
stopwords = f.read()
stopwords = stopwords.split('\n')
temp =[]

for i in range(len(data['Tweet'])):
    x = data['Tweet'][i].split()
    resultwords  = [word for word in x if word.lower() not in stopwords]
    data['Tweet'][i] = ' '.join(resultwords)
    
print(data)  


# In[32]:


total_vocab = [x for x in data['Tweet']]
bagofwords = []
total_count = 0
for i in total_vocab:
    tempwords = []
    tempwords.append(i.split())
    for x in tempwords:
        bagofwords.append(x)
    total_count += len(i.split()) 
print((total_vocab))


# In[8]:


def getUniqueWords(allWords) :
    uniqueWords = [] 
    for j in allWords:     
        for i in j:
            if not i in uniqueWords:
                uniqueWords.append(i)
    return uniqueWords
unique = getUniqueWords(bagofwords)
print(len(unique))


# In[9]:


bagofword1 = data['Tweet'][0]
bagofword1 = bagofword1.split(' ')
bagofword2 = data['Tweet'][1]
bagofword2 = bagofword2.split(' ')
print(bagofword2)


# In[10]:


word2count = {} 
for j in data['Tweet']:
    j = j.split()
    words = j
    for word in words:
            if word not in word2count.keys(): 
                word2count[word] = 1
            else: 
                word2count[word] += 1

total_words = [x for x in word2count]
print(len(total_words))


# In[11]:


new_myarray = []
key_object = {}
for j in range(len(data['Tweet'])):
    u = data['Tweet'][j]
    u = u.split()
    key_object[j] = u
new_arr = []
for i in key_object.values():
    new_arr.append(i)


# In[12]:



def func(bagofwords):
    numOfWordsA = dict.fromkeys(total_words, 0)
    for word in bagofwords:
        numOfWordsA[word] += 1
    return numOfWordsA
please = []
for h in range(len(new_arr)):
    d = func(new_arr[h])
    
    please.append(d)
func_data = pd.DataFrame(please)
func_data


# In[13]:


import collections 
tf_idf = {}
for j in data['Tweet']:
    j = j.split()
    tokens = j
    counter = collections.Counter(tokens + j)
    for token in np.unique(tokens):
        tf = counter[token]/len(total_words)
        df = word2count.get(token)
        idf = np.log(len(data['Tweet'])/(df+1))
        tf_idf[token] = tf*idf
tf_idf


# In[14]:


dictionary = func_data.iloc[0]
dictionary = pd.DataFrame(dictionary)
dictionary = dictionary.to_dict()
for x in dictionary.values():
    print(x)


# In[15]:


def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        
        tfidf[word] = val * idfs[word]
    return tfidf


# In[16]:


tfidfA = []
for i in range(len(please)):
    
    dictionary = func_data.iloc[i]
    dictionary = pd.DataFrame(dictionary)
    dictionary = dictionary.to_dict()
    for x in dictionary.values():
        p = x
    tfidfA.append(computeTFIDF(p, tf_idf))
df = pd.DataFrame(tfidfA)
df


# In[ ]:





# In[17]:


# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# tfidf_vect = TfidfVectorizer()
# corpus = total_vocab
# X1 = tfidf_vect.fit(corpus)
# # print(cv.get_feature_names())
# X1 = tfidf_vect.transform(corpus)
# # print(X1.shape)
# tf_data = pd.DataFrame(X1.toarray(), columns = tfidf_vect.get_feature_names())
# from IPython.display import display, HTML

# display(HTML((tf_data.head()).to_html()))


# In[18]:


# testData = pd.read_csv("Documents/Assignment 1/swad_test.csv")
# testData['Tweet'] = testData['Tweet'].str.lower()
# for i in punctuations:
#     testData['Tweet'] = testData['Tweet'].str.replace( i, " %s "%(i,), regex=True)

# for i in range(len(testData['Tweet'])):
#     x = testData['Tweet'][i].split()
#     resultwords  = [word for word in x if word.lower() not in stopwords]
#     testData['Tweet'][i] = ' '.join(resultwords)
    
# testData['Label'] = testData['Label'].replace({'No': 0, 'Yes': 1}).astype(int)
# total_vocab_test = [x for x in testData['Tweet']]


# tfidf_vect_test = TfidfVectorizer()
# corpus_test = total_vocab_test
# X1 = tfidf_vect_test.fit(corpus_test)
# # print(cv.get_feature_names())
# X1 = tfidf_vect_test.transform(corpus_test)
# # print(X1.shape)
# tf_data_test = pd.DataFrame(X1.toarray(), columns = tfidf_vect_test.get_feature_names())
# from IPython.display import display, HTML

# display(HTML((tf_data_test.head()).to_html()))


# In[19]:


X_train = df
X_train = X_train.values
data['Label'] =data['Label'].replace({'No': 0, 'Yes': 1}).astype(int)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train,data['Label'], test_size=0.25)
X_train = X_train.T
X_test = X_test.T
# Y_train = data['Label']
Y_train = Y_train.values
Y_train = pd.Series(Y_train)
Y_train = Y_train.values
Y_train = Y_train.reshape((1, X_train.shape[1]))
Y_test = Y_test.values
Y_test = pd.Series(Y_test)
Y_test = Y_test.values
Y_test = Y_test.reshape((1, X_test.shape[1]))
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:





# In[20]:



# Creating the Bag of Words model 

# word2count = {} 
# for j in data['Tweet']:
#     j = j.split()
#     words = j
#     for word in words:
#             if word not in word2count.keys(): 
#                 word2count[word] = 1
#             else: 
#                 word2count[word] += 1
# (word2count)


# In[21]:


# import heapq 

# freq_words = heapq.nlargest(500, word2count, key=word2count.get)


# In[22]:


# import nltk
# import numpy as np
# X = [] 
# for j in range(len(data['Tweet'])):
#     x = data['Tweet'][j].split()
#     vector = [] 
#     for word in freq_words: 
#         if word in x: 
#             vector.append(1) 
#         else: 
#             vector.append(0) 
#     X.append(vector) 
# XP_bow = np.asarray(X)
# len(XP_bow)


# In[ ]:





# In[23]:


# temp1 = data['Tweet']
# temp2 = data['Label']


# In[24]:


# X_train = data
# data['Label'] =data['Label'].replace({'No': 0, 'Yes': 1}).astype(int)
# testData = pd.read_csv("Documents/Assignment 1/swad_test.csv")
# testData['Tweet'] = testData['Tweet'].str.lower()
# for i in punctuations:
#     testData['Tweet'] = testData['Tweet'].str.replace( i, " %s "%(i,), regex=True)

# for i in range(len(testData['Tweet'])):
#     x = testData['Tweet'][i].split()
#     resultwords  = [word for word in x if word.lower() not in stopwords]
#     testData['Tweet'][i] = ' '.join(resultwords)
    
# testData['Label'] = testData['Label'].replace({'No': 0, 'Yes': 1}).astype(int)
# word2count_test = {} 

# for j in testData['Tweet']:
#     j = j.split()
#     words = j
#     for word in words: 
#             if word not in word2count_test.keys(): 
#                 word2count_test[word] = 1
#             else: 
#                 word2count_test[word] += 1
# freq_words_test = heapq.nlargest(500, word2count_test, key=word2count_test.get)
# T = []
# for j in range(len(testData['Tweet'])):
#     x = data['Tweet'][j].split()
#     vector_test = [] 
#     for word in freq_words_test: 
#         if word in x: 
#             vector_test.append(1) 
#         else: 
#             vector_test.append(0) 
#     T.append(vector_test) 
# XPtest_bow = np.asarray(T)
# print(len(XPtest_bow))
# print(testData)


# In[ ]:





# In[25]:



# Y_train = data['Label']
# Y_train = Y_train.values
# # X_train = XP_bow

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(XP_bow, data['Label'], test_size=0.28)
# print(X_train.shape)
# X_train = X_train.T
# # X_test = XPtest_bow
# X_test = X_test.T
# # Y_test = testData['Label']
# # Y_test = Y_test.values
# Y_train = pd.Series(Y_train)
# Y_train = Y_train.values
# Y_train = Y_train.reshape((1, X_train.shape[1]))
# Y_test = pd.Series(Y_test)
# Y_test = Y_test.values
# Y_test = Y_test.reshape((1, X_test.shape[1]))
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)


# In[26]:


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# In[27]:



def model(X, Y, learning_rate, iterations):
    
    m = X_train.shape[1]
    n = X_train.shape[0]
    
    W = np.zeros((n,1))
    B = 0
    
    cost_list = []
    
    for i in range(iterations):
        
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        
        # cost function
        cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
        
        # Gradient Descent
        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A - Y)
        
        W = W - learning_rate*dW.T
        B = B - learning_rate*dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list


# In[28]:


iterations = 10000
learning_rate = 0.0015
W, B, cost_list = model(X_train, Y_train, learning_rate = learning_rate, iterations = iterations)


# In[29]:


plt.plot(np.arange(iterations), cost_list)
plt.show()


# In[30]:


def accuracy(X, Y, W, B):
    print(Y)
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")


# In[31]:


accuracy(X_test, Y_test, W, B)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




