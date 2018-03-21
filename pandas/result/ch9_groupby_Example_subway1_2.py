
# coding: utf-8

# ## 예제: subway1
# 1. 각 노선번호 별 승차와 하차의 합 구하기
# 2. 각 노선번호별 시간에 따른 승차에 대한 증가추이와, 하차에 대한 증가추이를 하나의 피겨에 각각 생성
# 3. 지하철 운영 회사별 승하차에 대한 평균값 구하기
#   ** 서울메트로(2,4호선), 코레일(1,3호선), 도시철도공사(5,6,7,8)!

# In[84]:


from pandas import Series, DataFrame
import pandas as pd


# In[72]:


# %cd data
# %cd dataAnal_py_basic/pandas
# %cd ..


# In[88]:


subway= pd.read_csv('data/subway.csv', encoding='cp949')#, nrows=2)
subway.head(3)


# In[95]:


# subway= pd.read_csv('data/subway.csv', encoding='cp949'
#                    ,parse_dates=["시간"])
# subway.head(3)


# In[96]:


# type(subway['시간'])


# ### 1

# ### df [[ ]]. groupby( df [ ] )

# In[11]:


sub1ab= subway[['승차','하차']].groupby(subway['노선번호'])


# In[12]:


sub1ab.sum()


# In[14]:


sub1ab.sum().sort_values('승차', ascending=False)


# ### 2

# #### groupby는 당장 뭘 보는 용도로만 쓰고
# 이것 갖고 딴짓 할 생각 ㄴㄴ. 하려면 df 만들고 하던가

# In[95]:


import matplotlib.pyplot as plt


# In[126]:


fig, axes= plt.subplots(2,1)


# In[94]:


sub1t_on= subway[['승차','시간']].groupby(subway['노선번호'])
sub1t_off= subway[['하차','시간']].groupby(subway['노선번호'])


# In[153]:


subDicOn= dict(list(sub1t_on))
subDicOff= dict(list(sub1t_off))


# In[111]:


dict(list(sub1t_on))['line_1'].head(3)


# In[121]:


dict(list(sub1t_on)).keys()


# In[101]:


type(dict(list(sub1t_on))['line_1'])


# In[148]:


axes[1].plot(dict(list(sub1t_on))['line_1'])
fig


# In[154]:


fig.clear()
fig, axes= plt.subplots(2,1)


# In[150]:


axes[1].plot(dict(list(sub1t_on))['line_1'].set_index('시간'))
fig


# In[151]:


axes[1].plot(dict(list(sub1t_on))['line_2'].set_index('시간'))
fig


# In[163]:


fig.clear()
fig, axes= plt.subplots(2,1)


# In[164]:


list(subDicOn.keys())[0]


# subDicOn= dict(list(sub1t_on)), 
# subDicOff= dict(list(sub1t_off))

# axes[1].plot(dict(list(sub1t_on))['line_2'].set_index('시간'))
# fig

# In[165]:


lines= list(subDicOn.keys())


# In[166]:


for line in lines:
    axes[0].plot(subDicOn[line].set_index('시간'))
    axes[1].plot(subDicOff[line].set_index('시간'))
fig


# ### 위 방법의 문제점: 복잡하기도 하고, 범례가 알아서 안 그려진다
# plot은 어차피 정보 3개만 있으면 되니, pivot으로 하자

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[26]:


subway= pd.read_csv('data/subway.csv', encoding='cp949')#, nrows=2)
subway.head(3)


# In[27]:


subway.columns


# In[57]:


subP_on= subway.pivot('노선번호','시간','승차')


# In[59]:


subP_on


# In[28]:


subP_on_T= subway.pivot('노선번호','시간','승차').T
subP_off_T= subway.pivot('노선번호','시간','하차').T


# ### 시간이 행, 범례 대상이 컬럼으로 되어야!

# In[75]:


subP_on_T.head(3)


# In[80]:


# pd.to_datetime(subP_on_T.index)
type(subP_on_T.index)


# In[140]:


fig.clear('all')
fig, axes= plt.subplots(2,1)
fig.set_size_inches(10,7)


# In[141]:


#ㄴㄴ plt.rc('fig',figsize=(8,10))
plt.rc('font', family='Malgun Gothic')
plt.rc('legend', fontsize=10)


# In[73]:


# subP_on_T.plot(ax=axes[0])
# subP_off_T.plot(ax=axes[1])
# fig


# #### x축 hour, min을 좀더 알아보기 쉽게 바꾸고픔
# 05:30 ~ 12:30

# ### 일단 눈금 바꾸는 방법부터?
# - p315
# - axex[0].set_xticks

# In[93]:


# subP_on_T.plot?


# In[92]:


# ff= lambda x: ""
subP_on_T.index


# In[145]:


indN= ["%02d:%02d"%(i,j) for i,j in zip(range(5,25),range(6,25))]
indN.append('24:01')
indN


# In[150]:


type(indN)


# In[149]:


# int64index(indN)


# In[91]:


#인자가 숫자라 안됨 re.sub(subP_on_T.index,'\d{2}$','')


# In[153]:


type(xtick1)


# In[154]:


# 실행할때마다 범례가 늘어나는 에러가 있음. clear 해줘야!
subP_on_T.plot(ax=axes[0], xticks= xtick1)#, figsize=(8,10))
subP_off_T.plot(ax=axes[1])#, figsize=(8,10))#, xticks= indN)
fig


# In[54]:


axes[0].plot(subP_on_T)
axes[1].plot(subP_off_T)
fig


# ### seaborn으로도 해보자

# In[98]:


import seaborn as sns


# In[135]:


fig, axes= plt.subplots(3,1)
fig.set_size_inches(10,8)


# In[119]:


subway.head(3)


# In[124]:


get_ipython().magic('matplotlib inline')


# In[129]:


sns.barplot(data=subway, x="시간",y="승차", ax=axes[0])


# In[133]:


sns.pointplot(data=subway, x='시간', y='승차', ax=axes[1])


# In[136]:


axes[2].plot(subway, x='')


# In[138]:


get_ipython().set_next_input('axes[2].plot');get_ipython().magic('pinfo plot')


# In[ ]:


axes[2].plot


# #### sns.barplot?
# 인자 순서가 x, y, hue, data, ... 라서 명시해줘야 제대로 먹힘

# In[137]:


fig


# In[131]:


get_ipython().magic('pinfo pd.MultiIndex')


# ## 예제: subway2
# 1. 각 시간대 별 승차와 합차의 평균과 합을 나타내세요
# 2. 각 역별 전체 승차와 합차의 평균과 합을 나타내세요
# 3. 각 시간대 별 승차수가 가장 높은 역을 나타내세요
# 4. 승차가 가장 많은 top 10개의 역을 구하고 각 역의 시간대별 승차의 증감추세를 도표화 하여라
# 5. 각 역별 마지막 시간(24~01) 승차수와 첫시간(05~06) 승차수의 차이를 구하여라.

# In[62]:


get_ipython().system('start .')


# In[82]:


xls_file= pd.ExcelFile('data/subway2.xlsx')
# xls_file.sheet_names[0]
subwayy= xls_file.parse(xls_file.sheet_names[0], skiprows=1)


# In[116]:


subwayy


# In[134]:


subwayy= subwayy.fillna(method='ffill')
subwayy


# In[138]:


#df.set_index(['year', 'month'])
subwayy.set_index(['전체', '구분'])


# ### 1

# In[139]:


subwayy.groupby('구분').agg(['mean',sum])


# ### 2
# 이게 답이 아니네

# In[140]:


# dict(list(subwayy.groupby('전체')))
subwayy.groupby('전체').agg(['mean',sum])


# ### 3

# In[143]:


subwayy_si= subwayy.set_index(['전체', '구분'])
subwayy_si


# In[144]:


subwayy_si.max(0)


# In[146]:


subwayy_si.groupby('구분').agg(max)


# ### 4

# In[151]:


# import numpy as np
# subwayy_si.sort_values(by='구분')
# np.vectorize(top)(subwayy_si, 10, ???)


# In[179]:


from pandas import DataFrame


# In[176]:


# subwayy_si.index


# In[188]:


print(type(subwayy_si.sum(1)))
subwayy_si.sum(1)


# In[196]:


get_ipython().magic('pinfo DataFrame.xs')


# In[222]:


subwayy_si.sum(1).sort_values


# In[294]:


# subwayy_si.groupby(['전체','구분']).agg(sum)


# In[255]:


subwayy.sum(1)


# In[234]:


# subwayy.groupby('전체').apply(sum)


# In[274]:


subwayy[subwayy['구분']=='승차'].iloc[:,0:2]


# In[267]:


subwayy[subwayy['구분']=='승차'].sum(1)


# In[548]:


subwayy.set_index(['전체','구분']).T.unstack('승차')


# In[516]:


subwayy[subwayy['구분']=='하차'].iloc[:,0:2]


# In[277]:


subwayy2= subwayy[subwayy['구분']=='승차'].iloc[:,0:2]


# In[279]:


subwayy2['합']= subwayy[subwayy['구분']=='승차'].sum(1)


# In[280]:


subwayy2


# In[301]:


def top(df, n=10, column='합'):
    return df.sort_values(by=column, ascending=False)[:10]


# In[368]:


subTop10= top(subwayy2)
subTop10


# In[369]:


subwayy2.sort_values(by='합')[-10:]


# ### 4-2
# 다 그리는 게 아니라, 4-1 의 역만 추려서!

# In[315]:


subwayy[subwayy['구분']=='승차']


# In[319]:


subwayy[subwayy['구분']=='승차'].iloc[:,2:]


# In[307]:


import matplotlib.pyplot as plt


# In[309]:


get_ipython().magic('matplotlib inline')


# In[372]:


fig.clear('all')
fig, axes= plt.subplots(1)
fig.set_size_inches(20,10)


# In[331]:


subwayy[subwayy['구분']=='승차'].iloc[:,2:].plot(ax= axes)


# In[332]:


fig


# In[388]:


# subTop10.set_index('전체')
subTop10.set_index('전체').iloc[:,1]


# In[395]:


subTop10.전체


# In[391]:


fig.clear('all')
fig, axes= plt.subplots(1)
fig.set_size_inches(20,10)


# #### 이건 top10 역 승차수 비교 plot이고

# In[392]:


subTop10.set_index('전체').iloc[:,1].plot(ax=axes)


# In[393]:


fig


# In[362]:


subwayy_si.idxmax()


# In[367]:


subwayy.rank(ascending=False)['전체'].sort_values()


# In[343]:


print(subwayy[subwayy['구분']=='승차'].iloc[:,2:].T.shape)
print(subwayy['전체'].unique().shape)


# In[344]:


subwayy['전체'].unique()


# In[360]:


subwayy[subwayy['구분']=='승차'].iloc[:,2:]


# In[358]:


print(subwayy[subwayy['구분']=='승차'].iloc[:,2:].T.shape)


# In[493]:


subwayyToff= subwayy[subwayy['구분']=='하차'].iloc[:,2:].T
subwayyToff.columns= subwayy['전체'].unique()
subwayyToff


# In[346]:


subwayyT= subwayy[subwayy['구분']=='승차'].iloc[:,2:].T


# In[347]:


subwayyT.columns= subwayy['전체'].unique()


# In[348]:


subwayyT


# In[354]:


plt.rc('font', family='Malgun Gothic')


# In[502]:


fig.clear('all')
fig, axes= plt.subplots(1)
fig.set_size_inches(20,10)


# In[503]:


# subwayy[subwayy['구분']=='승차'].iloc[:,2:].T.plot(ax= axes)
subwayyT.plot(ax=axes)


# In[504]:


fig


# #### 전체 다 말고 top10역만;;

# In[402]:


from pandas import Series


# In[405]:


top10sta= list(subTop10.전체)
top10sta


# In[414]:


subwayyT[top10sta]


# In[487]:


plt.rc('font', family='Malgun Gothic')
plt.rc('legend', fontsize=30)


# In[511]:


fig.clear('all')
fig, axes= plt.subplots(2,1)
fig.set_size_inches(20,30)


# In[512]:


subwayyT[top10sta].plot(ax=axes[0], title='승차Top10')
subwayyToff[top10sta].plot(ax=axes[1], title='하차Top10')


# In[513]:


fig


# ### 5

# In[420]:


subwayyT


# In[424]:


subwayyT.xs('05~06')


# In[425]:


subwayyT.xs('24~01')


# ### 강사님 코드

# In[553]:


get_ipython().system('start .')


# In[568]:


a = pd.ExcelFile('data/subway2.xlsx', encoding='cp949')
raw= a.parse(a.sheet_names[0], skiprows=1)
raw.head(4)


# In[569]:


raw= raw.fillna(method='ffill')
raw.head(4)


# In[572]:


raw_si= raw.set_index(['전체','구분'])
raw_si.head(4)


# ### 1

# In[573]:


raw.groupby('구분').agg([sum,'mean'])


# ### 2

# In[575]:


raw_si.mean(axis=1)


# In[578]:


raw_si.sum(axis=1)


# ### 3

# In[603]:


data1= raw[raw['구분']=='승차'].iloc[:,2:]
data1
# could not convert string to float: '승차'


# In[697]:


data_off= raw[raw['구분']=='하차'].set_index('전체').iloc[:,1:]
data_off


# In[605]:


data1= raw[raw['구분']=='승차'].set_index('전체').iloc[:,1:]
data1


# In[606]:


data1.idxmax()


# In[699]:


data_off.idxmax()


# ### 4

# In[607]:


data1


# In[611]:


data1['total']= data1.sum(axis=1)
data1


# In[616]:


data1['total'].rank(ascending=False).sort_values()


# ### whos: 변수 뭐뭐 선언했었는지

# In[696]:


whos


# In[626]:


top10on= list(data1['total'].rank(ascending=False).sort_values()[:10].index)
top10on


# In[631]:


data1.loc[top10on][['total']]


# In[668]:


DataFrame(data1.iloc[:,-1]).loc[top10on]


# In[652]:


data1[['total']].loc[top10on]


# ### 4-2

# In[669]:


get_ipython().magic('matplotlib inline')


# In[671]:


data1.loc[top10on].iloc[:,:-2]


# In[674]:


data1T= data1.loc[top10on].iloc[:,:-2].T
data1T


# In[690]:


fig.clear('all')
fig, axis= plt.subplots(1)
fig.set_size_inches(20,20)
plt.rc('font', family='Malgun Gothic')
plt.rc('legend', fontsize= 20)


# In[691]:


data1T.plot(ax=axis)
fig


# ### 하차?
# 

# In[715]:


data_off= raw[raw['구분']=='하차'].set_index('전체').iloc[:,1:]
data_off


# In[716]:


data_off['total']= data_off.sum(axis=1)
data_off


# In[717]:


# top10on


# In[718]:


top10off= list(data_off['total'].rank(ascending=False).sort_values()[:10].index)
top10off


# In[719]:


data_off.loc[top10off][['total']]


# In[720]:


data_offT= data_off.loc[top10off].iloc[:,:-2].T
data_offT


# In[721]:


fig.clear('all')
fig, axis= plt.subplots(1)
fig.set_size_inches(20,20)
plt.rc('font', family='Malgun Gothic')
plt.rc('legend', fontsize= 20)


# In[722]:


data_offT.plot(ax=axis)
fig


# In[724]:


# %save subwayOffTop10.py 715-722


# ## 예제: kimchi
# kimchi_test.csv
# 
# ---
# 1. 각 년도별 제품별 판매량과 판매금액의 평균
# 2. 각 년도별 제품별 판매처별 판매량과 판매평균
#     1. groupby
#     2. pivot_table
# 3. 각 제품별 판매수량이 0~ 10000, 10000~20000, 20000~ 의 판매량을 기록한 월이 몇 회인지 확인

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


kch= pd.read_csv('data/kimchi_test.csv', encoding='cp949')


# In[15]:


print(kch.columns)
print(kch.shape)
kch.head()


# In[25]:


fig, axis= plt.subplots(2,1)
plt.rc('font', family= 'Malgun Gothic')
sns.barplot(data=kch, x="판매년도", y="수량", ax=axis[0])
sns.barplot(data=kch, x="판매년도", y="판매금액", ax=axis[1])
fig


# ### 1
# 각 년도별 제품별 판매량과 판매금액의 평균

# In[34]:


kch.pivot_table(index='판매년도', values=['수량','판매금액'])


# #### 평균이니 pivot_table이 편할것

# In[52]:


# # 위엣것이 더 보기 편함
# form_4= lambda x: "%.4f"%x
# kch.pivot_table(index='판매년도', values=['수량','판매금액']).applymap(form_4)


# ### 2 
# 각 년도별 제품별 판매처별 판매량과 판매평균
#     1. groupby
#     2. pivot_table

# ##### 2-1

# In[55]:


# 안됨 kch[['수량, 판매금액']].groupby(kch['판매년도'])


# #### 강사님코드

# In[57]:


kch.groupby(['판매년도','제품','판매처'])['수량','판매금액'].mean()


# In[60]:


kch.groupby(['판매년도','제품','판매처'])['수량','판매금액'].agg(sum)


# ##### 2-2

# In[48]:


kch.pivot_table(values=['수량'], index=['판매년도','제품','판매처'], margins=True)


# In[49]:


kch.pivot_table(values=['수량'], index=['판매년도'], columns=['제품','판매처'], margins=True)


# ### 강사님코드

# In[62]:


import numpy as np


# In[66]:


kch.pivot_table(values=['수량'], index=['판매년도'], columns=['제품','판매처'], aggfunc=[sum, np.mean])


# ### 3
# 각 제품별 판매수량이 0~ 10000, 10000~20000, 20000~ 의 판매량을 기록한 월이 몇 회인지 확인
# [pd.cut](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html)

# In[73]:


kch.수량.max()


# In[81]:


kch_ea= pd.cut(kch.수량, bins=[0, 10000, 20000, 70000])
kch_ea


# #### count니 crosstab이 편할것

# In[83]:


kch.columns


# In[84]:


pd.crosstab(kch_ea, kch.판매월, margins=True)


# In[85]:


# pd.crosstabl(g1.raw['제품'], margins=True)

