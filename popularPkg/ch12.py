
# coding: utf-8

# # Advanced pandas

# In[ ]:


import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)


# ## Categorical Data

# ### Background and Motivation

# In[ ]:


import numpy as np; import pandas as pd
values = pd.Series(['apple', 'orange', 'apple',
                    'apple'] * 2)
values
pd.unique(values)
pd.value_counts(values)


# In[ ]:


values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
values
dim


# In[ ]:


dim.take(values)


# ### Categorical Type in pandas

# In[ ]:


fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': np.random.randint(3, 15, size=N),
                   'weight': np.random.uniform(0, 4, size=N)},
                  columns=['basket_id', 'fruit', 'count', 'weight'])
df


# In[ ]:


fruit_cat = df['fruit'].astype('category')
fruit_cat


# In[ ]:


c = fruit_cat.values
type(c)


# In[ ]:


c.categories
c.codes


# In[ ]:


df['fruit'] = df['fruit'].astype('category')
df.fruit


# In[ ]:


my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
my_categories


# In[ ]:


categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(codes, categories)
my_cats_2


# In[ ]:


ordered_cat = pd.Categorical.from_codes(codes, categories,
                                        ordered=True)
ordered_cat


# In[ ]:


my_cats_2.as_ordered()


# ### Computations with Categoricals

# In[ ]:


np.random.seed(12345)
draws = np.random.randn(1000)
draws[:5]


# In[ ]:


bins = pd.qcut(draws, 4)
bins


# In[ ]:


bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
bins
bins.codes[:10]


# In[ ]:


bins = pd.Series(bins, name='quartile')
results = (pd.Series(draws)
           .groupby(bins)
           .agg(['count', 'min', 'max'])
           .reset_index())
results


# In[ ]:


results['quartile']


# #### Better performance with categoricals

# In[ ]:


N = 10000000
draws = pd.Series(np.random.randn(N))
labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))


# In[ ]:


categories = labels.astype('category')


# In[ ]:


labels.memory_usage()
categories.memory_usage()


# In[ ]:


get_ipython().magic("time _ = labels.astype('category')")


# ### Categorical Methods

# In[ ]:


s = pd.Series(['a', 'b', 'c', 'd'] * 2)
cat_s = s.astype('category')
cat_s


# In[ ]:


cat_s.cat.codes
cat_s.cat.categories


# In[ ]:


actual_categories = ['a', 'b', 'c', 'd', 'e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s2


# In[ ]:


cat_s.value_counts()
cat_s2.value_counts()


# In[ ]:


cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
cat_s3
cat_s3.cat.remove_unused_categories()


# #### Creating dummy variables for modeling

# In[ ]:


cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')


# In[ ]:


pd.get_dummies(cat_s)


# ## Advanced GroupBy Use

# ### Group Transforms and "Unwrapped" GroupBys

# In[ ]:


df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                   'value': np.arange(12.)})
df


# In[ ]:


g = df.groupby('key').value
g.mean()


# In[ ]:


g.transform(lambda x: x.mean())


# In[ ]:


g.transform('mean')


# In[ ]:


g.transform(lambda x: x * 2)


# In[ ]:


g.transform(lambda x: x.rank(ascending=False))


# In[ ]:


def normalize(x):
    return (x - x.mean()) / x.std()


# In[ ]:


g.transform(normalize)
g.apply(normalize)


# In[ ]:


g.transform('mean')
normalized = (df['value'] - g.transform('mean')) / g.transform('std')
normalized


# ### Grouped Time Resampling

# In[ ]:


N = 15
times = pd.date_range('2017-05-20 00:00', freq='1min', periods=N)
df = pd.DataFrame({'time': times,
                   'value': np.arange(N)})
df


# In[ ]:


df.set_index('time').resample('5min').count()


# In[ ]:


df2 = pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a', 'b', 'c'], N),
                    'value': np.arange(N * 3.)})
df2[:7]


# In[ ]:


time_key = pd.TimeGrouper('5min')


# In[ ]:


resampled = (df2.set_index('time')
             .groupby(['key', time_key])
             .sum())
resampled
resampled.reset_index()


# ## Techniques for Method Chaining

# ```python
# df = load_data()
# df2 = df[df['col2'] < 0]
# df2['col1_demeaned'] = df2['col1'] - df2['col1'].mean()
# result = df2.groupby('key').col1_demeaned.std()
# ```

# ```python
# # Usual non-functional way
# df2 = df.copy()
# df2['k'] = v
# 
# # Functional assign way
# df2 = df.assign(k=v)
# ```

# ```python
# result = (df2.assign(col1_demeaned=df2.col1 - df2.col2.mean())
#           .groupby('key')
#           .col1_demeaned.std())
# ```

# ```python
# df = load_data()
# df2 = df[df['col2'] < 0]
# ```

# ```python
# df = (load_data()
#       [lambda x: x['col2'] < 0])
# ```

# ```python
# result = (load_data()
#           [lambda x: x.col2 < 0]
#           .assign(col1_demeaned=lambda x: x.col1 - x.col1.mean())
#           .groupby('key')
#           .col1_demeaned.std())
# ```

# ### The pipe Method

# ```python
# a = f(df, arg1=v1)
# b = g(a, v2, arg3=v3)
# c = h(b, arg4=v4)
# ```

# ```python
# result = (df.pipe(f, arg1=v1)
#           .pipe(g, v2, arg3=v3)
#           .pipe(h, arg4=v4))
# ```

# ```python
# g = df.groupby(['key1', 'key2'])
# df['col1'] = df['col1'] - g.transform('mean')
# ```

# ```python
# def group_demean(df, by, cols):
#     result = df.copy()
#     g = df.groupby(by)
#     for c in cols:
#         result[c] = df[c] - g[c].transform('mean')
#     return result
# ```

# ```python
# result = (df[df.col1 < 0]
#           .pipe(group_demean, ['key1', 'key2'], ['col1']))
# ```

# In[ ]:


pd.options.display.max_rows = PREVIOUS_MAX_ROWS


# ## Conclusion
