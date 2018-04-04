
# coding: utf-8

# ### bikeShare
# [bikeShare](https://www.kaggle.com/c/bike-sharing-demand)
# text없고 encoding 되어 있어서 분석하는 맛이 있는 데이터. 대여 개수가 얼마나 되냐

# research engineer: 데이터를 잘 다룰 수 있게 정리해줌. 주로 개발쪽. 연구실에서 누가 대신해주면 좋겠음. 많은 사람이 필요하고, 많은 자리가 비어 있음. 석사가 힘들경우. pandas 깊게 파면 됨. 그분들은 이거 하는것도 귀찮아함

# regression problem: 정수형의 높고 낮음을 맞추기(ex. 주가)
# count 맞추기
# 
# // classif: 관련이 있는지

# 날짜 다루기

# evaluation 평가방식 배우기
# '%'는 직관적이나, data군이 9950 vs 50 명인 경우엔 좋은 게 아님
# 다 암환자가 아니라고 하면 99.5%이므로
# 
# 대신,

# In[75]:


(9950/9950+0/50)/2


# kaggle에 제출 않더라도
# 모델이 좋은지 아닌지 판단하는 방법

# In[76]:


import pandas as pd


# casual: 비 회원 대여량
# registered: 회원 대여량

# In[77]:


train= pd.read_csv("data/train.csv", parse_dates=["datetime"])
test= pd.read_csv("data/test.csv", parse_dates=["datetime"])


# parse_dates: 판다가 문자로 인식했던걸 날짜로 인식하게끔

# In[78]:


print(train.shape)
print(test.shape)
print(train.head())
print(test.head())


# atemp: 체감온도.
# datetime은 나중에 더 유용하게 쓸거고, 이거 외엔 index로 쓸만한 열이 없음

# In[79]:


train["datetime"].dt.year.head()
train["datetime"].dt.month.head()
train["datetime"].dt.second.head()


# In[80]:


train.head(1)


# 다 value라 preprossecing 이 필요 없음

# ### train

# In[81]:


feature_names=["season","holiday","workingday","weather","temp","atemp","humidity","windspeed"]
feature_names


# In[82]:


# X는 matrix같은, y는 vector같은 느낌이라 대소문자
# 근데 구글 최신 코드는 둘다 소문자 쓰더라


# In[83]:


X_train= train[feature_names]

print(X_train.shape)
X_train.head()


# In[84]:


X_test= test[feature_names]

print(X_test.shape)
X_test.head()


# In[85]:


label_name= "count"
y_train= train[label_name]

print(y_train.shape)
y_train.head()


# In[86]:


# from sklearn.tree import DecisionTreeRegressor
# # model= DecisionTreeRegressor()
# model= DecisionTreeRegressor(random_state=37) #모델개선 정도를 알아보려고 특정 랜덤을 고정시키는것

# model


# In[87]:


from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor(random_state=37) #모델개선 정도를 알아보려고 특정 랜덤을 고정시키는것

model


# ### score

# In[88]:


from sklearn.model_selection import cross_val_score
# score= cross_val_score(model, X_train, y_train, cv=20).mean() #cv 기본값은 3
score= cross_val_score(model, X_train, y_train, cv=20, scoring="neg_mean_absolute_error").mean() #0에 근접하도록 만듦

# score= -1*score

print("Score= {0:.5f}".format(score))
#0 에 근접할지만 알면 되고, +- 는 중요치 않아. sclearen의 역사랑 얽혀있는거


# X는 matrix같은, y는 vector같은 느낌이라 대소문자
# 

# In[89]:


model.fit(X_train, y_train)


# In[90]:


predictions= model.predict(X_test)
print(predictions.shape)
predictions[0:10]
#0시, 1시, ...


# ### submit

# In[91]:


#캐글 제출형식으로 만들기
submit= pd.read_csv("data/sampleSubmission.csv")
print(submit.shape)
submit.head()


# In[92]:


#예측값 넣기
submit["count"]= predictions
print(submit.shape)
submit.head()


# In[93]:


#파일 저장
# submit.to_csv("data/baseline_script.csv") # 이대로 넣으면 판다가 기본 인덱스 0~n 을 넣어버림
submit.to_csv("data/baseline_script.csv", index=False)


# decision tree: score: 1.62927
# #.
# random forest: 1.39409

# 값이 작을수록 좋은것

# 캐글에 제출 않고도

# #### 1.Hold-out validation
# train에서 또 쪼개서 일부는 train, 나머지는 test로 쓰고, 전체 fit값과 비교
# #####
# 단, 어떻게 쪼개느냐에 따라 결과가 많이 달라짐

# #### 2. Cross Validation
# train을 균등하게 쪼갬. 조각 ABCD중, 3개는 fit, 나머지는 predict해서 4번 함
# #####
# 캐글 결과랑 비슷한 결과. 단, 몹시 느림(특히 tree)
# 
# ##### 티탄 gpu 8개 박은걸로 4주 걸리는데, 이걸 cross V 하면 2달 걸림
# 잘게 쪼갤 수록 캐글이랑 가까워짐. 보통은 5~10조각 함. 강사님은 경험상 20조각이 좋았다시더라

# 데이터는 많을수록 좋으니, 쪼개서 한 fit model은 다 버리고 전체를 돌린다

# 판단기준이지만, 맹신 ㄴㄴ. 이게 떨어지는데 캐글값이 올라가는 경우도 있다

# ### 측정방식

# In[94]:


# actual 100
# model(a) 120
# model(b) 70


# ### 1. |p-a| : Mean absolute error(MAE)
# ### 2. (p-a)^2: Mean spanded error(MSE)
# 차이가 크면 클수록 패널티를 더 주고 싶을 때
# #.
# ex. 택시 기다리는 앱- 1~2분 차이는 ok. 5~10분 차이는 ㄴㄴ 

# ### 3. sqrt((p-a)^2): Root Mean Squared Error(RMSE)
# 2번과 유사. pycon 2017 발표하셨던거. 그냥 거의 똑같음 ㅇㅇ

# # 근데 bike에선 여기중 어떤것도 아닌, RMSLE란 기묘한 걸 씀- Overview 참조

# ### sqrt((log(p+1)-log(a+1))^2): RMSLE
# 개수가 많이 벌어져도 패널티가 적음

# pandas엔 없지만 직접구현? - 코드 공유해주실것

# ## Random Forrest: 데이터 일부를 샘플링하고 몹시 많은 tree를 만들고, 투표
# ### BAGGING
