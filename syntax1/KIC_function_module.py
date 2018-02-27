
# coding: utf-8

# ### Scope

# In[22]:


def fct1():
    """
    no param
    """
    a1= 10
    print("fct1 a1:",a1)


# In[23]:


def fct2():
    """
    no param
    """
    print("fct2 a1:",a1)


# In[24]:


a1= 20


# In[25]:


# del(a)


# ###### a10: 지역, a20: 전역

# In[26]:


fct1()
print(a1)
fct2()


# In[27]:


def fct1():
    global a
    a= 10
    print("fct1 a:",a)


# In[28]:


# 전역 a를 출력
def fct2():
    print("fct2 a:",a)


# ##### 순서주의!
# 1. a=20 뒤에 fct1() 하면 global a= 10 => a==10
# 2. fct1() 뒤에 전역 a= 20 하면 ==> a==20

# In[29]:


a=20


# In[30]:


#순서주의!
fct1() #여기서 global a= 10 됨
# a=20
print(a)
fct2()


# ##### keyword parameter
# ![image.png](attachment:image.png)

# In[31]:


def fct1(v1, v2= 0):
    """
    param1, param2
    """
    return v1+v2


# In[32]:


fct1(10)


# In[33]:


fct1(10,20)


# In[34]:


fct1(v1=10)


# In[35]:


# fct1(v1=10,20)


# In[36]:


fct1(v1=10, v2=20)


# ##### 가변 매개변수

# In[37]:


def fct_parm (*params):
    """
    *param
    """
    result= 0
    for param in params:
        result+= param
    return result


# In[38]:


print(fct_parm(10,20))


# In[39]:


print(fct_parm(10,20,30,40,50))


# ##### 키워드 매개변수2_dic

# In[40]:


def fct_dic(**param):
    """
    dictionary param: key= value, ...
    """
    for k in param.keys():
        print("%s---%s"%(k,param.get(k)))    


# In[41]:


fct_dic(트와이스=9, 소녀시대=7, 걸스데이=4, 블랙핑크=4)


# In[43]:


get_ipython().system('jupyter nbconvert --to script KIC_function_module.ipynb')


# In[44]:


#혹은 cmd에서
# jupyter nbconvert --to script KIC_function.ipynb

