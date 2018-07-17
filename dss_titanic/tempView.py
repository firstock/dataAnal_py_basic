# coding: utf-8
import pandas as pd
train= pd.read_csv("data/train.csv", index_col="PassengerId")
print(train.shape)
train.head()
import seaborn as sns
sns.countplot(data=train, x="Embarked", hue="Survived") 
sns.show()
