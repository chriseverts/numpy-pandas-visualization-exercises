#!/usr/bin/env python
# coding: utf-8

# In[236]:


import numpy as np

x = np.array([4, 10, 12, 23, -2, -1, 0, 0, 0, -6, 3, -7])

x


# In[237]:


##1. How many negative numbers are there?

x = np.array([4, 10, 12, 23, -2, -1, 0, 0, 0, -6, 3, -7])

import numpy as np

mask = x<0

neg_num = x[mask]

neg_num


# In[241]:


##2.How many positive numbers are there?

pos_num = (x > 0).sum()

print(pos_num)


# In[242]:


##3. How many even positive numbers are there?

x = np.array([4, 10, 12, 23, -2, -1, 0, 0, 0, -6, 3, -7])

even_num = (x[x % 2 == 0])

positive_even_num = even_num[even_num > 0].size



print(positive_even_num)


# In[247]:


##4.If you were to add 3 to each data point, how many positive numbers would there be?

x = np.array([4, 10, 12, 23, -2, -1, 0, 0, 0, -6, 3, -7])

add_num = (x + 3)

pos_num = add_num[add_num > 0]

print(pos_num)





# In[250]:


##5. If you squared each number, what would the new mean and standard deviation be?



square_num = (x**2).mean()
mean_num = np.mean(x)
std_num = (a ** 2).std()



print('The square is ',square_num)
print('The mean is ',mean_num)
print('The standard deviation is ',std_num)


# In[77]:


##6. A common statistical operation on a dataset is centering. 
#This means to adjust the data such that the mean of the data is 0.
#This is done by subtracting the mean from each data point. Center the data set. See this link for more on centering.

x = np.array([4, 10, 12, 23, -2, -1, 0, 0, 0, -6, 3, -7])

mean_num = np.mean(x)


center = x - mean_num

print(center)


# In[80]:


## 7. Calculate the z-score for each data point.

x = np.array([4, 10, 12, 23, -2, -1, 0, 0, 0, -6, 3, -7])

import pandas as pd
import numpy as np
import scipy.stats as stats

stats.zscore(x)



# In[81]:


import numpy as np
# Life w/o numpy to life with numpy

## Setup 1
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[84]:


# Exercise 1 - Make a variable called sum_of_a to hold the sum of all the numbers in above list


sum_of_a = np.sum(np.array(a))

print(sum_of_a)


# In[92]:


# Exercise 2 - Make a variable named min_of_a to hold the minimum of all the numbers in the above list


min_of_a = np.min(a)

print(min_of_a)


# In[91]:


# Exercise 3 - Make a variable named max_of_a to hold the max number of all the numbers in the above list


max_of_a = np.max(a)

print(max_of_a)



# In[90]:


# Exercise 4 - Make a variable named mean_of_a to hold the average of all the numbers in the above list


mean_of_a = np.mean(a)

print(mean_of_a)


# In[94]:


# Exercise 5 - Make a variable named product_of_a to hold the product of multiplying all the numbers in the above list together

product_of_a = np.prod(a)

print(product_of_a)


# In[95]:


# Exercise 6 - Make a variable named squares_of_a.

squares_of_a = np.square(a)

print(squares_of_a)



# In[110]:


# Exercise 7 - Make a variable named odds_in_a. It should hold only the odd numbers

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

odds_in_a = (a[a % 2 != 0])



print(odds_in_a)



# In[111]:


# Exercise 8 - Make a variable named evens_in_a. It should hold only the evens.


evens_in_a = (a[a % 2 == 0])



print(evens_in_a)




# In[121]:


## Setup 2: Consider what it would take to find the sum, min, max, average, sum, product, and list of squares for this list of two lists.

import numpy as np


b =np.array([
    [3, 4, 5],
    [6, 7, 8]])

sum_num = np.sum(b)
min_num = np.min(b)
max_num = np.max(b)
mean_num = np.mean(b)
product_num = np.prod(b)
square_num = np.square(b)

print(sum_num)
print(min_num)
print(max_num)
print(mean_num)
print(product_num)
print(square_num)


# In[122]:


# Exercise 1 - refactor the following to use numpy. Use sum_of_b as the variable.
##**Hint, you'll first need to make sure that the "b" variable is a numpy array**


b =np.array([
    [3, 4, 5],
    [6, 7, 8]])



sum_of_b = 0
for row in b:
    sum_of_b += sum(row)
    
print(sum_of_b)    
    
    
    
    
    
    
    


# In[133]:


# Exercise 2 - refactor the following to use numpy. 

min_of_b = min(b[0]) if min(b[0]) <= min(b[1]) else min(b[1])  


print(min_of_b)


# In[134]:


# Exercise 3 - refactor the following maximum calculation to find the answer with numpy.

max_of_b = max(b[0]) if max(b[0]) >= max(b[1]) else max(b[1])


print(max_of_b)


# In[135]:


# Exercise 4 - refactor the following using numpy to find the mean of b
mean_of_b = (sum(b[0]) + sum(b[1])) / (len(b[0]) + len(b[1]))

print(mean_of_b)


# In[136]:


# Exercise 5 - refactor the following to use numpy for calculating the product of all numbers multiplied together.
product_of_b = 1
for row in b:
    for number in row:
        product_of_b *= number
        
        
        
print(product_of_b)        


# In[137]:


# Exercise 6 - refactor the following to use numpy to find the list of squares 
squares_of_b = []
for row in b:
    for number in row:
        squares_of_b.append(number**2)

print(squares_of_b)


# In[139]:


# Exercise 7 - refactor using numpy to determine the odds_in_b
odds_in_b = []
for row in b:
    for number in row:
        if(number % 2 != 0):
            odds_in_b.append(number)
print(odds_in_b)


# In[140]:


# Exercise 8 - refactor the following to use numpy to filter only the even numbers
evens_in_b = []
for row in b:
    for number in row:
        if(number % 2 == 0):
            evens_in_b.append(number)
print(evens_in_b)


# In[234]:


# Exercise 9 - print out the shape of the array b.

b =np.array([
    [3, 4, 5],
    [6, 7, 8]])

shape = b.shape


print(shape)


# In[142]:


# Exercise 10 - transpose the array b.

num = np.transpose(b)

print(num)



# In[226]:


# Exercise 11 - reshape the array b to be a single list of 6 numbers. (1 x 6)

b =np.array([
    [3, 4, 5],
    [6, 7, 8]])



ray = np.arange(b)
b([:,None])

print(b)


# In[227]:


# Exercise 12 - reshape the array b to be a list of 6 lists, each containing only 1 number (6 x 1)




# In[229]:


## Setup 3

import numpy as np



c =np.array ([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

sum_num = np.sum(c)
min_num = np.min(c)
max_num = np.max(c)
product_num = np.prod(c)


print(sum_num)
print(min_num)
print(max_num)
print(product_num)



# HINT, you'll first need to make sure that the "c" variable is a numpy array prior to using numpy array methods.
# Exercise 1 - Find the min, max, sum, and product of c.


# In[231]:


# Exercise 2 - Determine the standard deviation of c.

std_num = np.std(c)

print(std_num)


# In[232]:


# Exercise 3 - Determine the variance of c.

import pandas as pd
import numpy as np
import scipy.stats as stats

stats.zscore(c)


# In[233]:


# Exercise 4 - Print out the shape of the array c


shape = c.shape

print(shape)




# In[235]:


# Exercise 5 - Transpose c and print out transposed result.

num = np.transpose(c)

print(num)



# In[ ]:




