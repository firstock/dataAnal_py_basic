# coding: utf-8
# %cd data
# %ls
# %cd ..
import pandas as pd
import matplotlib.pyplot as plt
employ= pd.read_csv('data/employment.csv', encoding='cp949', header=None)
print(employ.shape)
print(employ.columns)
employ.head(2)
# employ
empType= employ[0]
# 고용형태, 고용형태, 전체근로자, 전체근로자(특수형태포함)
#, 정규근로자, 비정규근로자, 비정규근로자(특수형태포함), 
#특수형태근로종사자, 재택/가내근로자, 파견/용역근로자, 
#일일근로자, 단시간근로자, 기간제근로자, 한시적근로자
empYear= employ.loc[0][1:]
#2007 9개, 2008 9개, ..., 2013 9개. 총63개
type(empYear)
employ.T.head(3)
# employ.T.reindex(empYear)
employ.T[0][0]= '연도'
empT= employ.T; empT
empT.to_csv('data/empT.csv', index_label=None)
emp= pd.read_csv('data/empT.csv', encoding='cp949', skiprows=1
                , usecols=range(1,15))
emp
type(emp.연도[1])
emp.set_index('연도')
emp.plot()
emp.columns[2]
emp_all= emp.pivot('연도','고용형태',emp.columns[2])
emp_all
# emp_all.plot(kind='bar')
# TypeError: Empty 'DataFrame': no numeric data to plot
print(type(emp_all.iloc[1][1]))
emp_all.iloc[1][1]
print(type(crime_happen.T.iloc[1][1]))
crime_happen.T.iloc[1][1]
crime_happen.T.head(2)
empY= emp.set_index('연도'); empY.head(3)
empY.applymap(np.isreal).head(3)
empY.iloc[:,1:]
# empY.iloc[:,1:].applymap(pd.to_numeric)
f1= lambda x: str.replace(x,',','')
f2= lambda x: str.replace(x,'-',str(0))
empY2= empY.applymap(f1).applymap(f2); empY2.head(8)
empY2_monSal= empY2[empY2.고용형태=='월급여액 (천원)']
empY2_monSal
empY3= empY2_monSal.iloc[:,1:].astype(float); empY3
empY3.plot(figsize= (8,12))
empY3.plot(kind='bar', title='월급여액 (천원)', figsize= (15,10), rot=45, fontsize=15)
empY3.T.plot(kind='bar', title='월급여액 (천원)', figsize=(15,15), rot=60, fontsize= 15)
