
# coding: utf-8

# # Data Loading, Storage, 

# In[ ]:


import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# ## Reading and Writing Data in Text Format

# In[ ]:


get_ipython().system('cat examples/ex1.csv')


# In[ ]:


df = pd.read_csv('examples/ex1.csv')
df


# In[ ]:


pd.read_table('examples/ex1.csv', sep=',')


# In[ ]:


get_ipython().system('cat examples/ex2.csv')


# In[ ]:


pd.read_csv('examples/ex2.csv', header=None)
pd.read_csv('examples/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])


# In[ ]:


names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('examples/ex2.csv', names=names, index_col='message')


# In[ ]:


get_ipython().system('cat examples/csv_mindex.csv')
parsed = pd.read_csv('examples/csv_mindex.csv',
                     index_col=['key1', 'key2'])
parsed


# In[ ]:


list(open('examples/ex3.txt'))


# In[ ]:


result = pd.read_table('examples/ex3.txt', sep='\s+')
result


# In[ ]:


get_ipython().system('cat examples/ex4.csv')
pd.read_csv('examples/ex4.csv', skiprows=[0, 2, 3])


# In[ ]:


get_ipython().system('cat examples/ex5.csv')
result = pd.read_csv('examples/ex5.csv')
result
pd.isnull(result)


# In[ ]:


result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])
result


# In[ ]:


sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('examples/ex5.csv', na_values=sentinels)


# ### Reading Text Files in Pieces

# In[ ]:


pd.options.display.max_rows = 10


# In[ ]:


result = pd.read_csv('examples/ex6.csv')
result


# In[ ]:


pd.read_csv('examples/ex6.csv', nrows=5)


# In[ ]:


chunker = pd.read_csv('examples/ex6.csv', chunksize=1000)
chunker


# In[ ]:


chunker = pd.read_csv('examples/ex6.csv', chunksize=1000)

tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)


# In[ ]:


tot[:10]


# ### Writing Data to Text Format

# In[ ]:


data = pd.read_csv('examples/ex5.csv')
data


# In[ ]:


data.to_csv('examples/out.csv')
get_ipython().system('cat examples/out.csv')


# In[ ]:


import sys
data.to_csv(sys.stdout, sep='|')


# In[ ]:


data.to_csv(sys.stdout, na_rep='NULL')


# In[ ]:


data.to_csv(sys.stdout, index=False, header=False)


# In[ ]:


data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])


# In[ ]:


dates = pd.date_range('1/1/2000', periods=7)
ts = pd.Series(np.arange(7), index=dates)
ts.to_csv('examples/tseries.csv')
get_ipython().system('cat examples/tseries.csv')


# ### Working with Delimited Formats

# In[ ]:


get_ipython().system('cat examples/ex7.csv')


# In[ ]:


import csv
f = open('examples/ex7.csv')

reader = csv.reader(f)


# In[ ]:


for line in reader:
    print(line)


# In[ ]:


with open('examples/ex7.csv') as f:
    lines = list(csv.reader(f))


# In[ ]:


header, values = lines[0], lines[1:]


# In[ ]:


data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict


# class my_dialect(csv.Dialect):
#     lineterminator = '\n'
#     delimiter = ';'
#     quotechar = '"'
#     quoting = csv.QUOTE_MINIMAL

# reader = csv.reader(f, dialect=my_dialect)

# reader = csv.reader(f, delimiter='|')

# with open('mydata.csv', 'w') as f:
#     writer = csv.writer(f, dialect=my_dialect)
#     writer.writerow(('one', 'two', 'three'))
#     writer.writerow(('1', '2', '3'))
#     writer.writerow(('4', '5', '6'))
#     writer.writerow(('7', '8', '9'))

# ### JSON Data

# In[ ]:


obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38,
               "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""


# In[ ]:


import json
result = json.loads(obj)
result


# In[ ]:


asjson = json.dumps(result)


# In[ ]:


siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
siblings


# In[ ]:


get_ipython().system('cat examples/example.json')


# In[ ]:


data = pd.read_json('examples/example.json')
data


# In[ ]:


print(data.to_json())
print(data.to_json(orient='records'))


# ### XML and HTML: Web Scraping

# conda install lxml
# pip install beautifulsoup4 html5lib

# In[ ]:


tables = pd.read_html('examples/fdic_failed_bank_list.html')
len(tables)
failures = tables[0]
failures.head()


# In[ ]:


close_timestamps = pd.to_datetime(failures['Closing Date'])
close_timestamps.dt.year.value_counts()


# #### Parsing XML with lxml.objectify

# <INDICATOR>
#   <INDICATOR_SEQ>373889</INDICATOR_SEQ>
#   <PARENT_SEQ></PARENT_SEQ>
#   <AGENCY_NAME>Metro-North Railroad</AGENCY_NAME>
#   <INDICATOR_NAME>Escalator Availability</INDICATOR_NAME>
#   <DESCRIPTION>Percent of the time that escalators are operational
#   systemwide. The availability rate is based on physical observations performed
#   the morning of regular business days only. This is a new indicator the agency
#   began reporting in 2009.</DESCRIPTION>
#   <PERIOD_YEAR>2011</PERIOD_YEAR>
#   <PERIOD_MONTH>12</PERIOD_MONTH>
#   <CATEGORY>Service Indicators</CATEGORY>
#   <FREQUENCY>M</FREQUENCY>
#   <DESIRED_CHANGE>U</DESIRED_CHANGE>
#   <INDICATOR_UNIT>%</INDICATOR_UNIT>
#   <DECIMAL_PLACES>1</DECIMAL_PLACES>
#   <YTD_TARGET>97.00</YTD_TARGET>
#   <YTD_ACTUAL></YTD_ACTUAL>
#   <MONTHLY_TARGET>97.00</MONTHLY_TARGET>
#   <MONTHLY_ACTUAL></MONTHLY_ACTUAL>
# </INDICATOR>

# In[ ]:


from lxml import objectify

path = 'datasets/mta_perf/Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()


# In[ ]:


data = []

skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
               'DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)


# In[ ]:


perf = pd.DataFrame(data)
perf.head()


# In[ ]:


from io import StringIO
tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()


# In[ ]:


root
root.get('href')
root.text


# ## Binary Data Formats

# In[ ]:


frame = pd.read_csv('examples/ex1.csv')
frame
frame.to_pickle('examples/frame_pickle')


# In[ ]:


pd.read_pickle('examples/frame_pickle')


# In[ ]:


get_ipython().system('rm examples/frame_pickle')


# ### Using HDF5 Format

# In[ ]:


frame = pd.DataFrame({'a': np.random.randn(100)})
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store


# In[ ]:


store['obj1']


# In[ ]:


store.put('obj2', frame, format='table')
store.select('obj2', where=['index >= 10 and index <= 15'])
store.close()


# In[ ]:


frame.to_hdf('mydata.h5', 'obj3', format='table')
pd.read_hdf('mydata.h5', 'obj3', where=['index < 5'])


# In[ ]:


os.remove('mydata.h5')


# ### Reading Microsoft Excel Files

# In[ ]:


xlsx = pd.ExcelFile('examples/ex1.xlsx')


# In[ ]:


pd.read_excel(xlsx, 'Sheet1')


# In[ ]:


frame = pd.read_excel('examples/ex1.xlsx', 'Sheet1')
frame


# In[ ]:


writer = pd.ExcelWriter('examples/ex2.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()


# In[ ]:


frame.to_excel('examples/ex2.xlsx')


# In[ ]:


get_ipython().system('rm examples/ex2.xlsx')


# ## Interacting with Web APIs

# In[ ]:


import requests
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
resp


# In[ ]:


data = resp.json()
data[0]['title']


# In[ ]:


issues = pd.DataFrame(data, columns=['number', 'title',
                                     'labels', 'state'])
issues


# ## Interacting with Databases

# In[ ]:


import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""
con = sqlite3.connect('mydata.sqlite')
con.execute(query)
con.commit()


# In[ ]:


data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()


# In[ ]:


cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows


# In[ ]:


cursor.description
pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


# In[ ]:


import sqlalchemy as sqla
db = sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test', db)


# In[ ]:


get_ipython().system('rm mydata.sqlite')


# ## Conclusion
