class PreProcessing:
    def __init__(self):
        pass

    @staticmethod
    def showCols():
        print("isnull: ", train[train["Embarked"].isnull()])
        cols1 = ["Sex", "Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]
        cols2 = ["Age", "Age_yy", "Age_y", "Age_o", "Age_oo", "Fare", "Fare_low", "Fare_mid", "Fare_high"]
        print("train: ", train[cols1 + cols2])
        print()
        print("test: ", test[cols1 + cols2])

    #         print((train.loc[[1],["Sex"]]=="male")|(train.loc[[1],["Sex"]]=="female"))

    def oneHotSex(self, df):
        # if (train.loc[[1],["Sex"]]=="male")|(train.loc[[1],["Sex"]]=="female"): #한번만 실행하려면??
        df.loc[df["Sex"] == "female", "Sex"] = 1
        df.loc[df["Sex"] == "male", "Sex"] = 0

    def oneHotEmbarked(self, df):
        cols = ["C", "S", "Q"]
        for col in cols:
            df["Embarked_" + col] = df["Embarked"] == col
            #         # 탑승지역으로 본 탑승국가. T/F
            #         df["Nation_FR"]= df["Embarked"]=="C"
            #         df["Nation_UK"]= df["Embarked"].isin(["S","Q"])

    def oneHotAge(self, df):
        # Age 별 나이대 나누기
        df["Age_yy"] = (df["Age"] < 11)
        df["Age_y"] = ((df["Age"] >= 11) & (df["Age"] < 20))
        df["Age_o"] = ((df["Age"] >= 20) & (df["Age"] < 50))
        df["Age_oo"] = (df["Age"] >= 50)

    def oneHotFare(self, df):
        # Fare 값 별 운임등급 나누기
        df["Fare_low"] = (df["Fare"] < 30)
        df["Fare_mid"] = ((df["Fare"] >= 30) & (df["Fare"] < 100))
        df["Fare_high"] = (df["Fare"] >= 100)


#
