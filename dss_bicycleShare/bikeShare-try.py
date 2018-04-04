
# coding: utf-8

# ### bikeShare
# [bikeShare](https://www.kaggle.com/c/bike-sharing-demand)
# text없고 encoding 되어 있어서 분석하는 맛이 있는 데이터. 대여 개수가 얼마나 되냐

# In[2]:


import pandas as pd


# casual: 비 회원 대여량
# registered: 회원 대여량

# In[3]:


train= pd.read_csv("data/train.csv", parse_dates=["datetime"])
test= pd.read_csv("data/test.csv", parse_dates=["datetime"])


# In[4]:


print(train.shape)
print(test.shape)
print(train.head())
print(test.head())


# In[5]:


train.loc[train["weather"].isin([1,2]),"weather12"]= 1
train.loc[train["weather"].isin([3,4]),"weather34"]= 0


# In[6]:


train["atemp_LL"]= (train["atemp"]<10) 
train["atemp_L"]= ((train["atemp"]>=10) & (train["atemp"]<20))
train["atemp_M"]= ((train["atemp"]>=20) & (train["atemp"]<30))
train["atemp_H"]= ((train["atemp"]>=30) & (train["atemp"]<40))
train["atemp_HH"]= (train["atemp"]>=40)


# In[7]:


test


# In[8]:


train["humidity10"]= floor(train["humidity"]/10)
test["humidity10"]= floor(test["humidity"]/10)


# In[9]:


# test엔 casual이 없다.
# train["regBigger"]= train["casual"] < train["registered"]
# test["regBigger"]= test["casual"] < test["registered"]


# atemp: 체감온도.
# datetime은 나중에 더 유용하게 쓸거고, 이거 외엔 index로 쓸만한 열이 없음

# In[10]:


train["datetime_Y"] = train["datetime"].dt.year
train["datetime_M"] = train["datetime"].dt.month
train["datetime_D"] = train["datetime"].dt.day
train["datetime_h"] = train["datetime"].dt.hour
train["datetime_m"] = train["datetime"].dt.minute
train["datetime_s"] = train["datetime"].dt.second

train["datetime_s"] = train["datetime"].dt.dayofweek


# In[11]:


train[["count","atemp_LL","atemp_L","atemp_M","atemp_H","atemp_HH"]]
train.to_csv("data/train_try4.csv", index=False)


# ### train

# In[14]:


# feature_names=["season","holiday","workingday","weather","temp","atemp","humidity","windspeed"]
feature_names=["season","holiday","workingday","weather","temp","atemp","humidity","windspeed"]

feature_names


# In[15]:


X_train= train[feature_names]

print(X_train.shape)
X_train.head()


# In[16]:


X_test= test[feature_names]

print(X_test.shape)
X_test.head()


# In[17]:


label_name= "count"
y_train= train[label_name]

print(y_train.shape)
y_train.head()


# In[18]:


from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor(random_state=37) #모델개선 정도를 알아보려고 특정 랜덤을 고정시키는것

model


# ### score

# In[19]:


from sklearn.model_selection import cross_val_score
# score= cross_val_score(model, X_train, y_train, cv=20).mean() #cv 기본값은 3
score= cross_val_score(model, X_train, y_train, cv=20, scoring="neg_mean_absolute_error").mean() #0에 근접하도록 만듦

# score= -1*score

print("Score= {0:.5f}".format(score))
#0 에 근접할지만 알면 되고, +- 는 중요치 않아. sclearen의 역사랑 얽혀있는거


# In[20]:


model.fit(X_train, y_train)


# In[21]:


predictions= model.predict(X_test)
print(predictions.shape)
predictions[0:10]
#0시, 1시, ...


# ### submit

# In[22]:


#캐글 제출형식으로 만들기
submit= pd.read_csv("data/sampleSubmission_try1.csv")
print(submit.shape)
submit.head()


# In[ ]:


#예측값 넣기
submit["count"]= predictions
print(submit.shape)
submit.head()


# In[ ]:


#파일 저장
# submit.to_csv("data/baseline_script.csv") # 이대로 넣으면 판다가 기본 인덱스 0~n 을 넣어버림
submit.to_csv("data/baseline_script.csv", index=False)


# decision tree: score: 1.62927
# #.
# random forest: 1.39409

# 값이 작을수록 좋은것

# 판단기준이지만, 맹신 ㄴㄴ. 이게 떨어지는데 캐글값이 올라가는 경우도 있다
