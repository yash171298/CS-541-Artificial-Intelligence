#!/usr/bin/env python
# coding: utf-8

# In[25]:


def twin_prime(n):
    prime = [True for i in range(n + 2)]
    p = 2
    q=0
    maximum_twin = []
    
    while (p * p <= n + 1):
         
        if (prime[p] == True):
             
            for i in range(p * 2, n + 2, p):
                prime[i] = False
        p += 1
        
    for p in range(2, n-1):
        if prime[p] and prime[p + 2]:
            maximum_twin.extend((p,p+2))
            print("(",p,",", (p + 2), ")" ,end='')
            q += 1
            
    print("Twin_prime are: ",q)        
    print("\nMaximum twin_prime is: ","(",maximum_twin[-2],",", maximum_twin[-1], ")" ,end='')
            
        
            
if __name__=='__main__':
    # static input
    try:
        n = int(input("Enter a number:"))
        assert n > 1
        twin_prime(n)
    except ValueError:
        print("The input provided was incorrect or not a number")
    except AssertionError:
        print("Provided Input should be positive or input has no twin primes between 1 to x")
        
    
    
     


# In[ ]:





# In[ ]:




