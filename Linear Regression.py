#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import mglearn
mglearn.plots.plot_linear_regression_wave()


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
lr=LinearRegression().fit(X_train,y_train)
print("lr.coef_:%s" % lr.coef_)
print("lr.intercept_:%s" % lr.intercept_)


# In[9]:


print("training set score:%f" % lr.score(X_train,y_train))
print("test set score:%f" % lr.score(X_test,y_test))


# In[ ]:




