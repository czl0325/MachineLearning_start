import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# 读取泰坦尼克号的所有乘客信息
data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
pd.set_option('display.max_columns', None)
# print(data.columns.values)
# survived字段为是否存活
x = data[["pclass", "age", "room", "sex"]]          # 特征值
y = data["survived"]                        # 目标值
# 把年纪的缺失值填补成x平均值
x.loc[:, "age"].fillna(x["age"].mean(), inplace=True)
x.loc[:, "room"].fillna("room", inplace=True)
print(x.isnull().any())

# 分割数据集到训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 进行处理（特征工程）特征->类别->one_hot编码   转化成字典
dict = DictVectorizer(sparse=False)
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
# print(dict.get_feature_names())
x_test = dict.transform(x_test.to_dict(orient="records"))

# 用决策树进行预测
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)
print("预测的准确率：", dec.score(x_test, y_test))
