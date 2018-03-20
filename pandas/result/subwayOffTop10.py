# coding: utf-8
data_off= raw[raw['구분']=='하차'].set_index('전체').iloc[:,1:]
data_off
data_off['total']= data_off.sum(axis=1)
data_off
# top10on
top10off= list(data_off['total'].rank(ascending=False).sort_values()[:10].index)
top10off
data_off.loc[top10off][['total']]
data_offT= data_off.loc[top10off].iloc[:,:-2].T
data_offT
fig.clear('all')
fig, axis= plt.subplots(1)
fig.set_size_inches(20,20)
plt.rc('font', family='Malgun Gothic')
plt.rc('legend', fontsize= 20)
data_offT.plot(ax=axis)
fig
