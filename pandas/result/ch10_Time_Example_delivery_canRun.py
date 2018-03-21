import pandas as pd
from dateutil.parser import parse
xlsx1= pd.ExcelFile('data/delivery_02.xlsx')
deliv2= xlsx1.parse(xlsx1.sheet_names[0], parse_dates=['일자'])
# deliv2

# ##### 2번_요일 칼럼을 만들자
deliv2['요일']= deliv2['일자'].dt.dayofweek
deliv2= deliv2.set_index('일자')
deliv2[deliv2.index=='2018-02-05'].head(1)

# #### 요일 0~6: 월~일
# 1. 2-05: 월요일
# 2. 월요일: 0
dow= [i for i in range(7)]; print(dow)
dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)
dic_dow= dict(zip(dow, dow_str)); dic_dow
deliv2['요일str']= deliv2['요일'].map(dic_dow)

# #### 3번_주중주말 칼럼을 만들자
deliv2.loc[deliv2["요일"]>=4,"주중주말"]= "주말"
deliv2.loc[deliv2["요일"]< 4,"주중주말"]= "주중"
print(deliv2[deliv2['주중주말']=='주중'].shape)
print(deliv2[deliv2['주중주말']=='주말'].shape)
# ,parse_dates=['일자'] 없이 그냥 데려왔을 때

# #### 날짜형식으로 바꾸기
# 1. pd.to_datetime(deliv['일자'], format='%Y%m%d')
# 2. deliv.일자= deliv.일자.apply(str).apply(parse)

# ### 1
# 일자별 총 통화건수와 전일대비 증가량!
# - 데이터 전체: 1달
# - resample에 인자 다 때려넣는 방식은 is deprecated

# ##### 일자별 총
deliv_D_call= deliv2[['통화건수']].resample('D').apply(sum)
deliv_D_call

# ##### 전일대비 증가량
deliv_D_call-deliv_D_call.shift(1)

# ### 2
# - 요일별로 각 업종별 통화건수를 확인
# - 막대그래프로 표현하여라
deliv2[['업종','통화건수']].resample('D').agg(sum)
# deliv2[['업종','통화건수']].resample('W-MON').agg(sum) #업종이 안나오는데다, 요일 하나만 나옴

# ##### 일단 업종별로 묶자
# 요일이 하나씩만 나옴
# deliv2[['업종','통화건수']].groupby(deliv2['업종']).resample('W-MON').agg(sum)
# 2_요일별로 각 업종별 통화건수를 확인하고 막대그래프로 표현
deliv2[['요일','업종','통화건수']]#.groupby('업종')

# #### 요일 0~6: 월~일
deliv_dow_job= deliv2[['요일str','업종','통화건수']].pivot_table(values='통화건수', index=['업종'], columns='요일str'
                                      ,aggfunc=sum)
deliv_dow_job
import matplotlib.pyplot as plt
# fig.clear('all')
plt.rc('font', family='Malgun Gothic')
fig, axis= plt.subplots(1)
fig.set_size_inches(10,10)
# deliv_dow_job.plot.bar(ax= axis)
deliv_dow_job.plot(kind='bar', ax= axis)
fig
fig.clear('all')
fig, axis= plt.subplots(1)
fig.set_size_inches(10,10)
deliv_dow_job.T.plot(kind='bar', ax= axis)
fig

# ### 3
# 평일과 주말(금,토,일)로 그룹을 나누어서, 각 그룹별 시군구별 통화건수를 분석

# #### 주중주말 칼럼을 만들자
# dic_dow
# dow= [i for i in range(7)]; print(dow)
# dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)
# deliv2.loc[deliv2["요일"]>=4,"주중주말"]= "주말"
# deliv2.loc[deliv2["요일"]< 4,"주중주말"]= "주중"
# print(deliv2[deliv2['주중주말']=='주중'].shape)
# print(deliv2[deliv2['주중주말']=='주말'].shape)
# deliv2
# 평일과 주말(금,토,일)로 그룹을 나누어서 각 그룹별 시군구별 통화건수를 분석하여라
deliv2[['주중주말','시군구','통화건수']].pivot_table(values='통화건수', index=['시군구'], columns='주중주말'
                                      ,aggfunc=sum)

# ### 4
# 시간대별로 가장 인기있는 음식업종을 출력하여라
# deliv2
pd.crosstab(deliv2.업종, deliv2.시간대)#.idxmax()
#idxmax: 칼럼별로 최고값을 갖는 인덱스값 반환
pd.crosstab(deliv2.업종, deliv2.시간대).idxmax(0)

# ### 5
# 그룹을 전달하면 

# ####  그룹내 통화건수가 가장 많은 상위 n개의 row가 출력되도록 
# 사용자 정의 함수를 만들고 적용
# # dataframe 전달하면
# deliv2.sort_values(by='통화건수', ascending=False).head(10)
# f3= lambda x: x.sort_values(by='통화건수', ascending=False).head(10)
# f3(deliv2)
# 말고, 꼭 groupby로 풀어야한다
# 그룹을 전달하면 

# ####  그룹내 통화건수가 가장 많은 상위 n개의 row가 출력되도록 
# 사용자 정의 함수를 만들고 적용
# import pandas as pd
# from dateutil.parser import parse
# xlsx1= pd.ExcelFile('data/delivery_02.xlsx')
# deliv2= xlsx1.parse(xlsx1.sheet_names[0], parse_dates=['일자'])
# # deliv2
# deliv2['요일']= deliv2['일자'].dt.dayofweek
# deliv2= deliv2.set_index('일자')
# dow= [i for i in range(7)]; print(dow)
# dow_str= ['MON','TUE','WED','THU','FRI','SAT','SUN']; print(dow_str)
# dic_dow= dict(zip(dow, dow_str)); dic_dow
# deliv2['요일str']= deliv2['요일'].map(dic_dow)
# deliv2.loc[deliv2["요일"]>=4,"주중주말"]= "주말"
# deliv2.loc[deliv2["요일"]< 4,"주중주말"]= "주중"
# deliv2
from pandas import DataFrame
deliv2.groupby(['시군구'])
DataFrame(deliv2.groupby(['시군구']).통화건수.agg(sum).sort_values(ascending=False)[:10])
from pandas import DataFrame
def groupTop10(g):
    return DataFrame((g).통화건수.agg(sum).sort_values(ascending=False)[:10])
groupTop10= lambda g: DataFrame((g).통화건수.agg(sum).sort_values(ascending=False)[:10])
group1= deliv2.groupby(['읍면동'])
groupTop10(group1)

# ### 아깐 되었는데 지금은 안 됨 ;;
# 커널 내렸다가 해도 안 됨
# 아깐 되었는데, 지금은 안 됨;
# from pandas import DataFrame
# def groupTop10(*g):
#     return (*g).통화건수.agg(sum).sort_values(ascending=False)[:10]
# group1= deliv2.groupby(['시군구','읍면동'])
# groupTop10(group1)
#여기서 시군구는 각 읍면동과 같은 행에 있어서 딸려나오는 것뿐
#신대방동 통화건수 합= 60049
#강서구 통화건수 합= 102926

# #### 순위는 rank 란 좋은 게 있다
#ex. 업종
deliv2.groupby(['시군구'])
deliv2.groupby(['시군구']).통화건수.agg(sum).rank()[:10]
import pandas
from pandas import DataFrame
# !jupyter nbconvert --to script ch10_Time_Example_delivery.ipynb
get_ipython().system('start .')