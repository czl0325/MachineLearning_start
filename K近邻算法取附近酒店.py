import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./csv/train.csv")
# 共29118021条数据，分6列，x,y代表坐标，accuracy为定位的精确度，time为定位时间，
# 以上几个都是特征值，目标值是place_id，定位的地点id
print(data.head(1))

# 由于29118021条数据太多，可以筛选去掉一部分，比如根据x和y来缩小
# 1、缩小数据,缩小后有17710条数据
data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")

# 2、处理时间的数据, unit代表精确到秒
time_value = pd.to_datetime(data["time"], unit='s')
# 把日期格式转换成 字典格式
time_value = pd.DatetimeIndex(time_value)

# 为数据补充day，hour，weekday三列数据
data.loc[:, "day"] = time_value.day
data.loc[:, "hour"] = time_value.hour
data.loc[:, "weekday"] = time_value.weekday

# 去掉原来time这一列，注意axis=1为列，=0为行
data = data.drop(["time"], axis=1)

# 找出出现次数大于3的地点
place_count = data.groupby("place_id").count()
tf = place_count[place_count.row_id > 3].reset_index()

# 得出签到数大于3次的地点列表
data = data[data["place_id"].isin(tf.place_id)]

# 取出数据当中的特征值和目标值
y = data["place_id"]  # 目标值
x = data.drop(["place_id", "row_id"], axis=1)  # 特征值(去掉place_id，row_id这两列)

# 进行数据的分割训练集合测试集(75%的训练集 25%的测试集)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 特征工程（标准化）
std = StandardScaler()
# 对测试集和训练集的特征值进行标准化
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# 进行k近邻算法流程
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# 得出预测结果
y_predict = knn.predict(x_test)
print("预测的目标签到位置为：", y_predict)

# 得出准确率
print("预测的准确率:", knn.score(x_test, y_test))
