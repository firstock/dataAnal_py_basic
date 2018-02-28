# release. https://codeshare.io/5QgEQm
# 수정 없을 시, 0.78947
# @site https://www.kaggle.com/c/titanic/leaderboard
class TreeSubmit:
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        predictions= []
        submit= []
    
    def wateringTree(self):
        featureNames= ["Pclass","Sex","Fare_low","Fare_mid","Fare_high","Embarked_C","Embarked_S","Embarked_Q"]
        label_name= "Survived"
        X_train= train[featureNames]
        y_train= train[label_name]
        
        model= DecisionTreeClassifier(max_depth=6)
        model.fit(X_train,y_train)
        
        X_test= test[featureNames]
        self.predictions= model.predict(X_test)
    
    # @param str
    def toSave(self, filename):
        self.submit= pd.read_csv("data/gender_submission.csv", index_col="PassengerId")
        self.submit["Survived"]= self.predictions
        self.submit.to_csv("data/"+filename+".csv")
        
    @staticmethod
    def showinfo():
        pass

#
#
#
#
#제일 마지막

ts= TreeSubmit()

ts.wateringTree()
ts.toSave("myty1120")