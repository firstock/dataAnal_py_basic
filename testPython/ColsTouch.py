# 열 만지작. 171119. https://codeshare.io/aJQRJR
# @date 171120
class ColsTouch:
    def __init__(self):  # 생성자? 겸 내부참조
        pass

    def show_info(self, dataframe=test, col="Age"):
        #     @staticmethod
        #     def show_info(dataframe,col):
        print("shape:", dataframe.shape)
        print("mean: ", dataframe[col].mean())
        print("count: ", dataframe[col].count())
        print("min: ", dataframe[col].min())
        print("max: ", dataframe[col].max())
        print("")

    def colAdd_Id(self, df):
        cnt = df.shape[0]
        df["Id"] = range(0, cnt)  # [0,cnt)

    def colAdd_Fam(self, df):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    def colAdd_Embarked(self, df):
        # 탑승국가 Nation. str
        df.loc[df["Embarked"] == "C", "Nation"] = "FR"
        df.loc[df["Embarked"].isin(["S", "Q"]), "Nation"] = "UK"

    def colAdd_Fare(self, df):
        low = (df["Fare"] < 30)
        mid = ((df["Fare"] >= 30) & (df["Fare"] < 100))
        high = (df["Fare"] >= 100)
        # Fare 등급
        df.loc[low, "FareGrade"] = "L"
        df.loc[mid, "FareGrade"] = "M"
        df.loc[high, "FareGrade"] = "H"

    def colMod_Fare(self, df):
        fare_mean = df["Fare"].mean()
        df.loc[df["Fare"].isnull(), "Fare"] = fare_mean

    def colAdd_Age(self, df):
        child = (df["Age"] < 11)
        adoles = ((df["Age"] >= 11) & (df["Age"] < 20))
        adult_y = ((df["Age"] >= 20) & (df["Age"] < 50))
        adult_o = (df["Age"] >= 50)
        # Age 세대
        df.loc[child, "AgeGen"] = "YY"
        df.loc[adoles, "AgeGen"] = "Y"
        df.loc[adult_y, "AgeGen"] = "O"
        df.loc[adult_o, "AgeGen"] = "OO"

    def colMod_Age(self, df):
        # NaN값 평균으로 채우기
        age_mean = df["Age"].mean()
        df.loc[df["Age"].isnull(), "Age"] = age_mean
