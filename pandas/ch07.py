
# coding: utf-8

# # Data Cleaning and Preparation

# In[ ]:


import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# ## Handling Missing Data

# In[ ]:


string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data
string_data.isnull()


# In[ ]:


string_data[0] = None
string_data.isnull()


# ### Filtering Out Missing Data

# In[ ]:


from numpy import nan as NA
data = pd.Series([1, NA, 3.5, NA, 7])
data.dropna()


# In[ ]:


data[data.notnull()]


# In[ ]:


data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                     [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
data
cleaned


# In[ ]:


data.dropna(how='all')


# In[ ]:


data[4] = NA
data
data.dropna(axis=1, how='all')


# In[ ]:


df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df
df.dropna()
df.dropna(thresh=2)


# ### Filling In Missing Data

# In[ ]:


df.fillna(0)


# In[ ]:


df.fillna({1: 0.5, 2: 0})


# In[ ]:


_ = df.fillna(0, inplace=True)
df


# In[ ]:


df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
df
df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)


# In[ ]:


data = pd.Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())


# ## Data Transformation

# ### Removing Duplicates

# In[ ]:


data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
data


# In[ ]:


data.duplicated()


# In[ ]:


data.drop_duplicates()


# In[ ]:


data['v1'] = range(7)
data.drop_duplicates(['k1'])


# In[ ]:


data.drop_duplicates(['k1', 'k2'], keep='last')


# ### Transforming Data Using a Function or Mapping

# In[ ]:


data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[ ]:


meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}


# In[ ]:


lowercased = data['food'].str.lower()
lowercased
data['animal'] = lowercased.map(meat_to_animal)
data


# In[ ]:


data['food'].map(lambda x: meat_to_animal[x.lower()])


# ### Replacing Values

# In[ ]:


data = pd.Series([1., -999., 2., -999., -1000., 3.])
data


# In[ ]:


data.replace(-999, np.nan)


# In[ ]:


data.replace([-999, -1000], np.nan)


# In[ ]:


data.replace([-999, -1000], [np.nan, 0])


# In[ ]:


data.replace({-999: np.nan, -1000: 0})


# ### Renaming Axis Indexes

# In[ ]:


data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])


# In[ ]:


transform = lambda x: x[:4].upper()
data.index.map(transform)


# In[ ]:


data.index = data.index.map(transform)
data


# In[ ]:


data.rename(index=str.title, columns=str.upper)


# In[ ]:


data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})


# In[ ]:


data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data


# ### Discretization and Binning

# In[ ]:


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]


# In[ ]:


bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats


# In[ ]:


cats.codes
cats.categories
pd.value_counts(cats)


# In[ ]:


pd.cut(ages, [18, 26, 36, 61, 100], right=False)


# In[ ]:


group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)


# In[ ]:


data = np.random.rand(20)
pd.cut(data, 4, precision=2)


# In[ ]:


data = np.random.randn(1000)  # Normally distributed
cats = pd.qcut(data, 4)  # Cut into quartiles
cats
pd.value_counts(cats)


# In[ ]:


pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])


# ### Detecting and Filtering Outliers

# In[ ]:


data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()


# In[ ]:


col = data[2]
col[np.abs(col) > 3]


# In[ ]:


data[(np.abs(data) > 3).any(1)]


# In[ ]:


data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()


# In[ ]:


np.sign(data).head()


# ### Permutation and Random Sampling

# In[ ]:


df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler


# In[ ]:


df
df.take(sampler)


# In[ ]:


df.sample(n=3)


# In[ ]:


choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
draws


# ### Computing Indicator/Dummy Variables

# In[ ]:


df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data1': range(6)})
pd.get_dummies(df['key'])


# In[ ]:


dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy


# In[ ]:


mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('datasets/movielens/movies.dat', sep='::',
                       header=None, names=mnames)
movies[:10]


# In[ ]:


all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)


# In[ ]:


genres


# In[ ]:


zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns=genres)


# In[ ]:


gen = movies.genres[0]
gen.split('|')
dummies.columns.get_indexer(gen.split('|'))


# In[ ]:


for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1


# In[ ]:


movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.iloc[0]


# In[ ]:


np.random.seed(12345)
values = np.random.rand(10)
values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))


# ## String Manipulation

# ### String Object Methods

# In[ ]:


val = 'a,b,  guido'
val.split(',')


# In[ ]:


pieces = [x.strip() for x in val.split(',')]
pieces


# In[ ]:


first, second, third = pieces
first + '::' + second + '::' + third


# In[ ]:


'::'.join(pieces)


# In[ ]:


'guido' in val
val.index(',')
val.find(':')


# In[ ]:


val.index(':')


# In[ ]:


val.count(',')


# In[ ]:


val.replace(',', '::')
val.replace(',', '')


# ### Regular Expressions

# In[ ]:


import re
text = "foo    bar\t baz  \tqux"
re.split('\s+', text)


# In[ ]:


regex = re.compile('\s+')
regex.split(text)


# In[ ]:


regex.findall(text)


# In[ ]:


text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)


# In[ ]:


regex.findall(text)


# In[ ]:


m = regex.search(text)
m
text[m.start():m.end()]


# In[ ]:


print(regex.match(text))


# In[ ]:


print(regex.sub('REDACTED', text))


# In[ ]:


pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)


# In[ ]:


m = regex.match('wesm@bright.net')
m.groups()


# In[ ]:


regex.findall(text)


# In[ ]:


print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))


# ### Vectorized String Functions in pandas

# In[ ]:


data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
data
data.isnull()


# In[ ]:


data.str.contains('gmail')


# In[ ]:


pattern
data.str.findall(pattern, flags=re.IGNORECASE)


# In[ ]:


matches = data.str.match(pattern, flags=re.IGNORECASE)
matches


# In[ ]:


matches.str.get(1)
matches.str[0]


# In[ ]:


data.str[:5]


# In[ ]:


pd.options.display.max_rows = PREVIOUS_MAX_ROWS


# ## Conclusion
