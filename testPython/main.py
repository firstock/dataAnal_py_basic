# https://codeshare.io/5XWEBz
# data 다운로드 https://www.kaggle.com/c/titanic/data
import pandas as pd
import ColsTouch
import PreProcessing
import TreeSubmit

train = pd.read_csv("data/train.csv", index_col="PassengerId")
test = pd.read_csv("data/test.csv", index_col="PassengerId")
print(train.shape)
print(test.shape)

#
#
#    첫 번째 실행
#
ct = ColsTouch()

ct.colMod_Age(train)
ct.colMod_Age(test)

ct.colMod_Fare(train)
ct.colMod_Fare(test)

# ct.colAdd_Age(train)
# ct.colAdd_Age(test)

# ct.colAdd_Id(train)
# ct.colAdd_Fam(train)
# ct.colAdd_Fare(train)
# ct.colAdd_Embarked(train)

# cols1= ["Id","FamilySize","Embarked","Nation","Fare","FareGrade"]
# cols2=["Age","AgeGen"]
# print(train[cols2].head(10))
# print(train.head(10))

# pp.show_info()
# pp.show_info(train,"Age")

# cols=["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]
# print(train[cols].head(8))


#
#
#
# import pandas as pd
# train= pd.read_csv("data/train.csv",index_col="PassengerId")
# test= pd.read_csv("data/test.csv",index_col="PassengerId")
# print(train.shape)
# print(test.shape)

#
#
# 두 번째 실행
#
pp = PreProcessing()

# 한 번만 실행 가능
pp.oneHotSex(train)
pp.oneHotSex(test)

pp.oneHotEmbarked(train)
pp.oneHotEmbarked(test)

# pp.oneHotAge(train)
# pp.oneHotAge(test)

# pp.oneHotFare(train)
# pp.oneHotFare(test)


#
#
#
# 세 번째 실행

ts = TreeSubmit()

ts.wateringTree()
ts.toSave("myty1120")