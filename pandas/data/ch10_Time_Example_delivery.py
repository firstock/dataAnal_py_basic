
# coding: utf-8

# # Time Series
# ## 시계열
# p393

# In[3]:


# !jupyter nbconvert --to script result/timeExamDeliv.ipynb


# In[4]:


get_ipython().system('start .')


# ## 예제: delivery
# 1. 일자별 총 통화건수와 전일대비 증가량을 구하여라
# 2. 요일별로 각 업종별 통화건수를 확인하고 막대그래프로 표현하여라
# 3. 평일과 주말(금,토,일)로 그룹을 나누어서 각 그룹별 시군구별 통화건수를 분석하여라
# 4. 시간대별로 가장 인기있는 음식업종을 출력하여라
# 5. 그룹을 전달하면 그룹내 통화건수가 가장 많은 상위 n개의 row가 출력되도록 사용자 정의 함수를 만들고 적용시켜보자

# In[86]:


import pandas as pd
from dateutil.parser import parse


# In[79]:


xlsx1= pd.ExcelFile('data/delivery_02.xlsx')


# In[297]:


deliv= xlsx1.parse(xlsx1.sheet_names[0])
deliv


# In[298]:


deliv.columns


# #### 날짜형식으로 바꾸기 3

# In[305]:


# 안됨 deliv.일자= pd.to_datetime(deliv.일자); deliv
pd.to_datetime(deliv['일자'], format='%Y%m%d')


# In[102]:


from dateutil.parser import parse


# In[112]:


print(type(deliv['일자'][0]))
parse(str(deliv['일자'][0]))


# In[115]:


deliv.일자= deliv.일자.apply(str).apply(parse)


# In[124]:


deliv= deliv.set_index('일자')
deliv


# In[135]:


deliv.index


# ### 1
# 일자별 총 통화건수와 전일대비 증가량!

# In[141]:


#데이터가 전부해서 1달
# deliv[['통화건수']].resample('M', how=sum)# 다더하는걸 원하는게 X


# ##### resample에 인자 다 때려넣는 방식은 is deprecated

# ##### 일자별 총

# In[143]:


deliv_D_call= deliv[['통화건수']].resample('D').apply(sum)
deliv_D_call


# ##### 전일대비 증가량

# In[145]:


deliv_D_call-deliv_D_call.shift(1)


# ### 2
# - 요일별로 각 업종별 통화건수를 확인
# - 막대그래프로 표현하여라

# In[152]:


deliv_D_call


# In[156]:


deliv[['업종','통화건수']].resample('D').agg(sum)


# In[158]:


deliv[['업종','통화건수']].resample('W-MON').agg(sum)


# ##### 일단 업종별로 묶자

# In[162]:


deliv[['업종','통화건수']].groupby(deliv['업종']).resample('W-MON').agg(sum)


# ##### 요일 칼럼을 만들자

# In[189]:


deliv2= xlsx1.parse(xlsx1.sheet_names[0], parse_dates=['일자'])
deliv2


# In[190]:


type(deliv2.일자[0])


# ##### 2-05: 월요일
# 월요일: 0

# In[195]:


deliv2[deliv2.일자=='2018-02-05'].head(1)


# In[191]:


deliv2['요일']= deliv2['일자'].dt.dayofweek


# In[192]:


deliv2


# In[196]:


#아님 from datetime.datetime import strptime
#안됨 deliv2[['일자']].apply(str).apply(strptime, '%Y-%m-%d-%w')


# In[198]:


deliv2= deliv2.set_index('일자')
deliv2


# 2. 요일별로 각 업종별 통화건수를 확인하고 막대그래프로 표현

# In[201]:


deliv2[['요일','업종','통화건수']]#.groupby('업종')


# #### 먼저 요일 0~6: 월~일 로 바꿔주고

# In[214]:


dow= [i for i in range(7)]; print(dow)
dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)


# In[216]:


dic_dow= dict(zip(dow, dow_str)); dic_dow


# In[222]:


deliv2['요일str']= deliv2['요일'].map(dic_dow)


# In[223]:


deliv2


# In[205]:


from pandas import DataFrame
import pandas as pd


# In[225]:


deliv_dow_job= deliv2[['요일str','업종','통화건수']].pivot_table(values='통화건수', index=['업종'], columns='요일str'
                                      ,aggfunc=sum)
deliv_dow_job


# In[227]:


import matplotlib.pyplot as plt


# In[269]:


fig.clear('all')
plt.rc('font', family='Malgun Gothic')
fig, axis= plt.subplots(1)


# In[270]:


# plt.plot(deliv_dow_job)
deliv_dow_job.plot.bar(ax= axis)


# In[267]:


deliv_dow_job.plot(kind='bar', ax= axis)


# In[271]:


fig


# ### 3
# 평일과 주말(금,토,일)로 그룹을 나누어서, 각 그룹별 시군구별 통화건수를 분석

# In[272]:


deliv2


# In[273]:


dic_dow


# In[214]:


dow= [i for i in range(7)]; print(dow)
dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)


# In[281]:


deliv2.loc[deliv2["요일"]>=4,"주중주말"]= "주말"
deliv2.loc[deliv2["요일"]< 4,"주중주말"]= "주중"


# In[287]:


get_ipython().magic('pinfo deliv2.loc')


# In[285]:


print(deliv2[deliv2['주중주말']=='주중'].shape)
print(deliv2[deliv2['주중주말']=='주말'].shape)


# In[288]:


deliv2


# 평일과 주말(금,토,일)로 그룹을 나누어서 각 그룹별 시군구별 통화건수를 분석하여라

# In[289]:


deliv2[['주중주말','시군구','통화건수']].pivot_table(values='통화건수', index=['시군구'], columns='주중주말'
                                      ,aggfunc=sum)


# ### 4
# 시간대별로 가장 인기있는 음식업종을 출력하여라

# In[290]:


deliv2


# In[294]:


deliv2[['시간대','업종']]
#.pivot_table(values='통화건수', index=['시군구'], columns='주중주말',aggfunc='count')

