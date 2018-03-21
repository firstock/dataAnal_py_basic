
# coding: utf-8

# ### [Pandas API](https://pandas.pydata.org/pandas-docs/stable/api.html#id7)

# In[28]:


from pandas import DataFrame, Series
import pandas as pd
import numpy as np


# In[7]:


print(pd.period_range('2014-10-18','2018-03-04', freq='D'))
len(pd.period_range('2014-10-18','2018-03-04', freq='D'))


# #### .infer_freq

# In[11]:


df= pd.read_table('./data/ex_df.txt')
df


# In[13]:


columns1=['name', 'price', 'qty']
df.columns= columns1
df


# In[15]:


index1= ['one','one','two']
df.index= index1
df


# In[22]:


# 쓰는 방법?? pd.infer_freq(df.index)


# In[24]:


ser1= Series([4,7,-5,3]); ser1


# In[25]:


ser1.ftype


# In[35]:


ser2= Series(np.eye(20)[1]); ser2


# In[36]:


ser2.ftype


# In[37]:


Series(np.eye(20)[1]).ftype


# In[41]:


df1= DataFrame([0,1,2,np.nan,4], columns=['B']); df1


# In[45]:


df1.ewm(com= 0.5).mean()


# #### pandas.Series.truncate

# In[48]:


chr(ord('a')+1)


# In[74]:


abcx= [chr(i) for i in range(97,97+24,1)]
arr_abcx= np.array(abcz).reshape((4,6))
# index1= range(1,5,1)
index1= ['A','G','M','S']
df_abcx= DataFrame(arr_abcx, index= index1)
df_abcx


# In[58]:


type(np.arange(5))


# In[77]:


df_abcx.truncate(before='G', after='M')


# In[78]:


df_abcx


# ## .query
# indexing의 왕인듯. 근데 evil 같단 느낌이..

# In[80]:


from numpy.random import randn
from pandas import DataFrame
df= DataFrame(randn(10,2), columns= list('ab'));df


# In[81]:


df.query('a>b')


# In[85]:


df.query('a.min()>b')


# In[83]:


df[df['a']>df['b']]

