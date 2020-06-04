import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import time

start = time.perf_counter()
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

# 分割数据集到训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 进行处理（特征工程）特征->类别->one_hot编码   转化成字典
dict = DictVectorizer(sparse=False)
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
# print(dict.get_feature_names())
x_test = dict.transform(x_test.to_dict(orient="records"))

# 随机森林
rf = RandomForestClassifier()
param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
gc = GridSearchCV(rf, param_grid=param, cv=10)
gc.fit(x_train, y_train)
print("准确率：", gc.score(x_test, y_test))
print("最佳参数：", gc.best_params_)

end = time.perf_counter()
print('程序运行时间: %s 秒' % (end-start))

"""
准确率： 0.7993920972644377
最佳参数： {'max_depth': 8, 'n_estimators': 300}
程序运行时间: 289.626128219 秒
"""