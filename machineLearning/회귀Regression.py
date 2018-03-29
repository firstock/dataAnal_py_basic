
# coding: utf-8

# # 회귀Regression

# 1. (다중) 선형Linear 회귀(최소제곱법) 
#     - 선 하나로 뭉뚱그리는 게 부정확해 보일 수 있지만,다차원 데이터 일 수록(식2갠데 변수 6개) 꽤 좋은 선택지
# 
# 2. 릿지Ridge 회귀
#     - 규제Regularization 중, L2
#         - 과대적합 방지. 가중치를 작게작게
#         - 차원 축소가 일어나지 않음
#     - 매개변수 alpha
#         - 값이 높을 수록 많이 단순해진다. 기본값 1
#         - 10쯤 넣으면 계수가 0에 가까워짐. 0이 되진 않음
#         - 단순해지면? 당장 모델(훈련)점수는 별로겠지만, 일반화에는 좋음
#         - 0.1 넣으면 산개되고_데이터 생김새에 가깝게. 요러면 선형회귀랑 거의 비슷
#         - 보통 10까지 넣음
#     - 제일 많이 씀
# 
# 3. 라쏘
#     - 규제 중, L1
#     - 가중치가 0까지 감. 해당 변수를 아예 제거하는 셈 ㅇㅇ

# ### 3개 한방에?
# 그냥 릿지 alpha값을 0.01~100 까지 돌리면 다 해보는 거 아님??
# > 아님 ㅇㅇ

# #### 모델 검증_예측일 때
# 모델 재료말고 다른걸로 테스트한 점수가 높은 게 좋은 것

# ##### 문제의식: 보스턴 data에
# 설명변수가 너무 많아서 KNN regression으로는 안 되었다. 선형회귀는 어떨까?

# ## 선형 회귀
# LinearRegression()

# ### 단순 선형회귀

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[29]:


import mglearn
from sklearn.model_selection import train_test_split


# In[30]:


X,y= mglearn.datasets.make_wave(n_samples=50)


# In[31]:


X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=42)


# In[32]:


lr= LinearRegression().fit(X_train, y_train)


# #### 결과확인
# 점수(결정계수)자체가 낮으니 좋은 모델은 아님. 다만, 값이 비슷하이 과대적합은 아님

# In[33]:


lr.coef_


# In[34]:


lr.intercept_


# In[35]:


print("훈련세트 점수: {:.5f}".format(lr.score(X_train, y_train)))
print("새로온 테스트: {:.5f}".format(lr.score(X_test, y_test)))


# ### 다중 선형회귀

# In[ ]:


X,y= mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=0)


# In[39]:


lr= LinearRegression().fit(X_train, y_train)


# In[40]:


print(X.shape, y.shape,'\n',X_train.shape, y_train.shape,'\n',  X_test.shape,y_test.shape)


# ##### 점수확인
# 훈련점수만 너무 높고, 새로온 건 별로. 과대적합 ㅇㅇ

# In[44]:


print("훈련 점수: %8.5f"%lr.score(X_train, y_train))
print("새로온 거: %8.5f"%lr.score(X_test, y_test))


# ## 릿지 Ridge 회귀
# 과대적합이 줄고, 새로온거 점수가 좋아졌다.
# > Boston() data에 대해선, 선형회귀보다 릿지가 좋다

# In[45]:


from sklearn.linear_model import Ridge


# In[47]:


X,y= mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=0)


# In[48]:


ridge= Ridge().fit(X_train, y_train)


# In[51]:


print("훈련 점수: %8.5f"%ridge.score(X_train, y_train))
print("새로온 거: %8.5f"%ridge.score(X_test, y_test))


# #### 그치만 0.75점이 높은 건 아니므로, alpha값 조정
# 1. alpha=10: 과대적합은 줄었지만, 새로온거 점수가 더 낮아졌다 ;;
# 2. alpha=0.1:과대적합이 폭발!! 새로온거 점수는 기본값 1보단 높아짐

# In[50]:


ridge10= Ridge(alpha=10).fit(X_train, y_train)


# In[52]:


print("훈련 점수:%8.5f"%ridge10.score(X_train,y_train))
print("새로온 거:%8.5f"%ridge10.score(X_test,y_test))


# In[53]:


ridge01= Ridge(alpha=.1).fit(X_train, y_train)


# In[55]:


print("훈련 점수:%8.5f"%ridge01.score(X_train,y_train))
print("새로온 거:%8.5f"%ridge01.score(X_test,y_test))


# #### 그래프: 각 alpha값에 따라 릿지 회귀계수 비교
# 1. Ridge alpha 10: 거의 0
# 2. Ridge alpha 1: 조금 산개. 조금 복잡
# 3. Ridge alpha 0.1: 좀더 산개. 좀더 복잡
# 4. Linear Regression: 완전 산개. 제일 복잡

# In[58]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[60]:


plt.rc('font', family='Malgun Gothic')


# In[61]:


plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="Linear Regression")

plt.xlabel('계수목록')
plt.ylabel('계수크기')
plt.hlines(0,0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()


# ## 라쏘 Lasso

# In[68]:


from sklearn.linear_model import Lasso
import numpy as np


# In[63]:


X,y= mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=0)


# In[74]:


print(X_train.shape,  y_train.shape, '\n',X_test.shape, y_test.shape)


# #### alpha값 변화에 따른 점수
# 1. 1 기본값: 점수가 ㅆㄺ같이 나왔다. 104개 중에 특성(칼럼) 4개는 너무 적음(복잡도가 너무 낮다)
# 2. 0.01: 점수가 크게 오름! 다만 훈련점수 대비 새로온거 점수가 낮다. 과적합
# 3. 0.0001: 훈련점수는 더욱 과적합되었고, 새로온거 점수는 더 낮아졌다

# In[79]:


lasso= Lasso().fit(X_train, y_train)


# In[80]:


print("훈련 점수:%8.5f"%lasso.score(X_train,y_train))
print("새로온 거:%8.5f"%lasso.score(X_test,y_test))

print("사용한특성 수:%4d"%np.sum(lasso.coef_ != 0))


# > ? max_iter가 뭐하는 값인지?

# In[85]:


#Objective did not converge. You might want to increase the number of iterations.
# Fitting data with very small alpha may cause precision problems. ConvergenceWarning)
# lasso001= Lasso(alpha=0.01).fit(X_train, y_train)
lasso01= Lasso(alpha=.01, max_iter=100000).fit(X_train, y_train)


# In[86]:


print("훈련 점수:%8.5f"%lasso01.score(X_train,y_train))
print("새로온 거:%8.5f"%lasso01.score(X_test,y_test))

print("사용한특성 수:%4d"%np.sum(lasso01.coef_ != 0))


# In[87]:


lasso0001= Lasso(alpha=.0001, max_iter=100000).fit(X_train, y_train)


# In[88]:


print("훈련 점수:%8.5f"%lasso0001.score(X_train,y_train))
print("새로온 거:%8.5f"%lasso0001.score(X_test,y_test))

print("사용한특성 수:%4d"%np.sum(lasso0001.coef_ != 0))


# Lasso alpha값이 낮아질수록 Linear Regression에 가까워지고 있다![image.png](attachment:image.png)

# ### Lasso: alpha값에 따른 비교
# 1. Lasso alpha값이 0에 가까울수록, 산개(복잡도가 높음. 각 데이터를 따라감)

# In[96]:


plt.plot(lasso.coef_, 's', label='Lasso alpha=1')
plt.plot(lasso01.coef_, '^', label='Lasso alpha=.01')
plt.plot(lasso0001.coef_, 'v', label='Lasso alpha=.0001')

plt.plot(ridge01.coef_, 'o', label='Ridge alpha=.1')

plt.legend(ncol=2, loc= (0, 1.05))
plt.ylim(-25, 25)
plt.xlabel('계수 목록')
plt.ylabel('계수 크기')


# In[ ]:


get_ipython().system('jupyter -nbconverter --py ')

