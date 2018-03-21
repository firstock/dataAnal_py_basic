
# coding: utf-8

# # Time Series
# ## 시계열
# p393

# In[11]:


# %ls


# In[109]:


# !jupyter nbconvert --to script ch10_Time_Example_delivery.ipynb


# In[13]:


# !start .
# get_ipython().system('start .')


# ## 예제: delivery
# 1. 일자별 총 통화건수와 전일대비 증가량을 구하여라
# 2. 요일별로 각 업종별 통화건수를 확인하고 막대그래프로 표현하여라
# 3. 평일과 주말(금,토,일)로 그룹을 나누어서 각 그룹별 시군구별 통화건수를 분석하여라
# 4. 시간대별로 가장 인기있는 음식업종을 출력하여라
# 5. 그룹을 전달하면 그룹내 통화건수가 가장 많은 상위 n개의 row가 출력되도록 사용자 정의 함수를 만들고 적용시켜보자

# In[14]:


import pandas as pd
from dateutil.parser import parse


# In[15]:


xlsx1= pd.ExcelFile('data/delivery_02.xlsx')


# In[16]:


deliv= xlsx1.parse(xlsx1.sheet_names[0])
deliv


# In[17]:


deliv.columns


# #### 날짜형식으로 바꾸기 3

# In[18]:


# 안됨 deliv.일자= pd.to_datetime(deliv.일자); deliv
pd.to_datetime(deliv['일자'], format='%Y%m%d')


# In[19]:


from dateutil.parser import parse


# In[20]:


print(type(deliv['일자'][0]))
parse(str(deliv['일자'][0]))


# In[21]:


deliv.일자= deliv.일자.apply(str).apply(parse)


# In[22]:


deliv= deliv.set_index('일자')
deliv


# In[23]:


deliv.index


# ### 1
# 일자별 총 통화건수와 전일대비 증가량!

# In[24]:


#데이터가 전부해서 1달
# deliv[['통화건수']].resample('M', how=sum)# 다더하는걸 원하는게 X


# ##### resample에 인자 다 때려넣는 방식은 is deprecated

# ##### 일자별 총

# In[25]:


deliv_D_call= deliv[['통화건수']].resample('D').apply(sum)
deliv_D_call


# ##### 전일대비 증가량

# In[26]:


deliv_D_call-deliv_D_call.shift(1)


# ### 2
# - 요일별로 각 업종별 통화건수를 확인
# - 막대그래프로 표현하여라

# In[27]:


deliv_D_call


# In[28]:


deliv[['업종','통화건수']].resample('D').agg(sum)


# In[29]:


deliv[['업종','통화건수']].resample('W-MON').agg(sum)


# ##### 일단 업종별로 묶자

# In[30]:


deliv[['업종','통화건수']].groupby(deliv['업종']).resample('W-MON').agg(sum)


# ##### 요일 칼럼을 만들자

# In[3]:


deliv2= xlsx1.parse(xlsx1.sheet_names[0], parse_dates=['일자'])
deliv2


# In[32]:


type(deliv2.일자[0])


# ##### 2-05: 월요일
# 월요일: 0

# In[33]:


deliv2[deliv2.일자=='2018-02-05'].head(1)


# In[34]:


deliv2['요일']= deliv2['일자'].dt.dayofweek


# In[35]:


deliv2


# In[36]:


#아님 from datetime.datetime import strptime
#안됨 deliv2[['일자']].apply(str).apply(strptime, '%Y-%m-%d-%w')


# In[37]:


deliv2= deliv2.set_index('일자')
deliv2


# 2. 요일별로 각 업종별 통화건수를 확인하고 막대그래프로 표현

# In[38]:


deliv2[['요일','업종','통화건수']]#.groupby('업종')


# #### 먼저 요일 0~6: 월~일 로 바꿔주고

# In[39]:


dow= [i for i in range(7)]; print(dow)
dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)


# In[40]:


dic_dow= dict(zip(dow, dow_str)); dic_dow


# In[41]:


deliv2['요일str']= deliv2['요일'].map(dic_dow)


# In[42]:


deliv2


# In[43]:


from pandas import DataFrame
import pandas as pd


# In[44]:


deliv_dow_job= deliv2[['요일str','업종','통화건수']].pivot_table(values='통화건수', index=['업종'], columns='요일str'
                                      ,aggfunc=sum)
deliv_dow_job


# In[45]:


import matplotlib.pyplot as plt


# In[126]:


fig.clear('all')
plt.rc('font', family='Malgun Gothic')
fig, axis= plt.subplots(1)
fig.set_size_inches(10,10)


# In[48]:


# plt.plot(deliv_dow_job)
deliv_dow_job.plot.bar(ax= axis)


# In[127]:


deliv_dow_job.plot(kind='bar', ax= axis)


# In[128]:


fig


# In[124]:


deliv_dow_job.T.plot(kind='bar', ax= axis)


# In[125]:


fig


# ### 3
# 평일과 주말(금,토,일)로 그룹을 나누어서, 각 그룹별 시군구별 통화건수를 분석

# In[51]:


deliv2


# In[52]:


dic_dow


# In[53]:


dow= [i for i in range(7)]; print(dow)
dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)


# In[54]:


deliv2.loc[deliv2["요일"]>=4,"주중주말"]= "주말"
deliv2.loc[deliv2["요일"]< 4,"주중주말"]= "주중"


# In[56]:


print(deliv2[deliv2['주중주말']=='주중'].shape)
print(deliv2[deliv2['주중주말']=='주말'].shape)


# In[57]:


deliv2


# 평일과 주말(금,토,일)로 그룹을 나누어서 각 그룹별 시군구별 통화건수를 분석하여라

# In[58]:


deliv2[['주중주말','시군구','통화건수']].pivot_table(values='통화건수', index=['시군구'], columns='주중주말'
                                      ,aggfunc=sum)


# ### 4
# 시간대별로 가장 인기있는 음식업종을 출력하여라

# In[59]:


deliv2


# In[74]:


#이건 그냥 시간대별 업종 수나 다름 없고
deliv2[['시간대','업종']].groupby(['시간대']).count()


# In[84]:


pd.crosstab(deliv2.업종, deliv2.시간대)#.idxmax()


# In[94]:


#idxmax: 칼럼별로 최고값을 갖는 인덱스값 반환
pd.crosstab(deliv2.업종, deliv2.시간대).idxmax(0)


# ### 5

# 그룹을 전달하면 
# ####  그룹내 통화건수가 가장 많은 상위 n개의 row가 출력되도록 
# 사용자 정의 함수를 만들고 적용

# In[103]:


deliv2.sort_values(by='통화건수', ascending=False).head(10)


# In[104]:


f3= lambda x: x.sort_values(by='통화건수', ascending=False).head(10)


# In[107]:


f3(deliv2)


# In[108]:


f3(deliv)


# 말고, 꼭 groupby로 풀어야한다

# 그룹을 전달하면 
# ####  그룹내 통화건수가 가장 많은 상위 n개의 row가 출력되도록 
# 사용자 정의 함수를 만들고 적용

# In[1]:


import pandas as pd
from dateutil.parser import parse
xlsx1= pd.ExcelFile('data/delivery_02.xlsx')
deliv2= xlsx1.parse(xlsx1.sheet_names[0], parse_dates=['일자'])
# deliv2

deliv2['요일']= deliv2['일자'].dt.dayofweek
deliv2= deliv2.set_index('일자')

dow= [i for i in range(7)]; print(dow)
dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)
dic_dow= dict(zip(dow, dow_str)); dic_dow
deliv2['요일str']= deliv2['요일'].map(dic_dow)

deliv2.loc[deliv2["요일"]>=4,"주중주말"]= "주말"
deliv2.loc[deliv2["요일"]< 4,"주중주말"]= "주중"


# In[2]:


deliv2


# In[8]:


from pandas import DataFrame


# In[9]:


#ex. 업종
deliv2.groupby(['시군구'])
DataFrame(deliv2.groupby(['시군구']).통화건수.agg(sum).sort_values(ascending=False)[:10])


# In[10]:


from pandas import DataFrame

def groupTop10(g):
    #return g.통화건수.agg(sum).sort_values(ascending=False)[:10]
    return DataFrame((g).통화건수.agg(sum).sort_values(ascending=False)[:10])


# In[13]:


groupTop10= lambda g: DataFrame((g).통화건수.agg(sum).sort_values(ascending=False)[:10])


# In[14]:


group1= deliv2.groupby(['읍면동'])
groupTop10(group1)


# ### 아깐 되었는데 지금은 안 됨 ;;
# 커널 내렸다가 해도 안 됨

# In[12]:


from pandas import DataFrame

def groupTop10(*g):
    return (*g).통화건수.agg(sum).sort_values(ascending=False)[:10]


# In[26]:


#신대방동 통화건수 합= 60049
#강서구 통화건수 합= 102926
group1= deliv2.groupby(['시군구']) #여기서 시군구는 각 읍면동과 같은 행에 있어서 딸려나오는 것뿐
# group1= deliv2.groupby(['시군구']) 
groupTop10(group1)


# #### 순위는 rank 란 좋은 게 있다

# In[203]:


#ex. 업종
deliv2.groupby(['시군구'])
deliv2.groupby(['시군구']).통화건수.agg(sum).rank()[:10]


# In[20]:


import pandas
from pandas import DataFrame


# In[15]:


# !jupyter nbconvert --to script ch10_Time_Example_delivery.ipynb


# In[16]:


get_ipython().system('start .')

