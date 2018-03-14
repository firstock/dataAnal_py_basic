
# coding: utf-8

# # Data Wrangling: Join, Combine, 

# In[ ]:


import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# ## Hierarchical Indexing

# In[ ]:


data = pd.Series(np.random.randn(9),
                 index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data


# In[ ]:


data.index


# In[ ]:


data['b']
data['b':'c']
data.loc[['b', 'd']]


# In[ ]:


data.loc[:, 2]


# In[ ]:


data.unstack()


# In[ ]:


data.unstack().stack()


# In[ ]:


frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                              ['Green', 'Red', 'Green']])
frame


# In[ ]:


frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame


# In[ ]:


frame['Ohio']


# MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']],
#                        names=['state', 'color'])

# ### Reordering and Sorting Levels

# In[ ]:


frame.swaplevel('key1', 'key2')


# In[ ]:


frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)


# ### Summary Statistics by Level

# In[ ]:


frame.sum(level='key2')
frame.sum(level='color', axis=1)


# ### Indexing with a DataFrame's columns

# In[ ]:


frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                      'c': ['one', 'one', 'one', 'two', 'two',
                            'two', 'two'],
                      'd': [0, 1, 2, 0, 1, 2, 3]})
frame


# In[ ]:


frame2 = frame.set_index(['c', 'd'])
frame2


# In[ ]:


frame.set_index(['c', 'd'], drop=False)


# In[ ]:


frame2.reset_index()


# ## Combining and Merging Datasets

# ### Database-Style DataFrame Joins

# In[ ]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
df1
df2


# In[ ]:


pd.merge(df1, df2)


# In[ ]:


pd.merge(df1, df2, on='key')


# In[ ]:


df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
                    'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')


# In[ ]:


pd.merge(df1, df2, how='outer')


# In[ ]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                    'data2': range(5)})
df1
df2
pd.merge(df1, df2, on='key', how='left')


# In[ ]:


pd.merge(df1, df2, how='inner')


# In[ ]:


left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                     'key2': ['one', 'two', 'one'],
                     'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                      'key2': ['one', 'one', 'one', 'two'],
                      'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')


# In[ ]:


pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))


# ### Merging on Index

# In[ ]:


left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                      'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
left1
right1
pd.merge(left1, right1, left_on='key', right_index=True)


# In[ ]:


pd.merge(left1, right1, left_on='key', right_index=True, how='outer')


# In[ ]:


lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio',
                               'Nevada', 'Nevada'],
                      'key2': [2000, 2001, 2002, 2001, 2002],
                      'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
                      index=[['Nevada', 'Nevada', 'Ohio', 'Ohio',
                              'Ohio', 'Ohio'],
                             [2001, 2000, 2000, 2000, 2001, 2002]],
                      columns=['event1', 'event2'])
lefth
righth


# In[ ]:


pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
pd.merge(lefth, righth, left_on=['key1', 'key2'],
         right_index=True, how='outer')


# In[ ]:


left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])
left2
right2
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)


# In[ ]:


left2.join(right2, how='outer')


# In[ ]:


left1.join(right1, on='key')


# In[ ]:


another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                       index=['a', 'c', 'e', 'f'],
                       columns=['New York', 'Oregon'])
another
left2.join([right2, another])
left2.join([right2, another], how='outer')


# ### Concatenating Along an Axis

# In[ ]:


arr = np.arange(12).reshape((3, 4))
arr
np.concatenate([arr, arr], axis=1)


# In[ ]:


s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])


# In[ ]:


pd.concat([s1, s2, s3])


# In[ ]:


pd.concat([s1, s2, s3], axis=1)


# In[ ]:


s4 = pd.concat([s1, s3])
s4
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join='inner')


# In[ ]:


pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])


# In[ ]:


result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result
result.unstack()


# In[ ]:


pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])


# In[ ]:


df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                   columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                   columns=['three', 'four'])
df1
df2
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])


# In[ ]:


pd.concat({'level1': df1, 'level2': df2}, axis=1)


# In[ ]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])


# In[ ]:


df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
df1
df2


# In[ ]:


pd.concat([df1, df2], ignore_index=True)


# ### Combining Data with Overlap

# In[ ]:


a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
a
b
np.where(pd.isnull(a), b, a)


# In[ ]:


b[:-2].combine_first(a[2:])


# In[ ]:


df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
                    'b': [np.nan, 2., np.nan, 6.],
                    'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
                    'b': [np.nan, 3., 4., 6., 8.]})
df1
df2
df1.combine_first(df2)


# ## Reshaping and Pivoting

# ### Reshaping with Hierarchical Indexing

# In[ ]:


data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))
data


# In[ ]:


result = data.stack()
result


# In[ ]:


result.unstack()


# In[ ]:


result.unstack(0)
result.unstack('state')


# In[ ]:


s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2
data2.unstack()


# In[ ]:


data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)


# In[ ]:


df = pd.DataFrame({'left': result, 'right': result + 5},
                  columns=pd.Index(['left', 'right'], name='side'))
df
df.unstack('state')


# In[ ]:


df.unstack('state').stack('side')


# ### Pivoting “Long” to “Wide” Format

# In[ ]:


data = pd.read_csv('examples/macrodata.csv')
data.head()
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter,
                         name='date')
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D', 'end')
ldata = data.stack().reset_index().rename(columns={0: 'value'})


# In[ ]:


ldata[:10]


# In[ ]:


pivoted = ldata.pivot('date', 'item', 'value')
pivoted


# In[ ]:


ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]


# In[ ]:


pivoted = ldata.pivot('date', 'item')
pivoted[:5]
pivoted['value'][:5]


# In[ ]:


unstacked = ldata.set_index(['date', 'item']).unstack('item')
unstacked[:7]


# ### Pivoting “Wide” to “Long” Format

# In[ ]:


df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                   'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})
df


# In[ ]:


melted = pd.melt(df, ['key'])
melted


# In[ ]:


reshaped = melted.pivot('key', 'variable', 'value')
reshaped


# In[ ]:


reshaped.reset_index()


# In[ ]:


pd.melt(df, id_vars=['key'], value_vars=['A', 'B'])


# In[ ]:


pd.melt(df, value_vars=['A', 'B', 'C'])
pd.melt(df, value_vars=['key', 'A', 'B'])


# ## Conclusion
