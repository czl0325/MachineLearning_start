import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 逻辑回归做二分类进行癌症预测
"""
   1. Sample code number            id number
   2. Clump Thickness               1 - 10      块厚度
   3. Uniformity of Cell Size       1 - 10      细胞大小的一致性
   4. Uniformity of Cell Shape      1 - 10      细胞形状的均匀性
   5. Marginal Adhesion             1 - 10      边缘附着力
   6. Single Epithelial Cell Size   1 - 10      单个上皮细胞的大小
   7. Bare Nuclei                   1 - 10      裸核
   8. Bland Chromatin               1 - 10      单调的染色质
   9. Normal Nucleoli               1 - 10      正常核仁
  10. Mitoses                       1 - 10      核分裂
  11. Class:                        (2 for benign, 4 for malignant)
"""
# 构造列标签名字 因为数据集没有标签行，需要手动指定标签行
column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column)
# 缺失值处理
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()

# 拆分数据
x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]])

# 标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# 逻辑回归预测
logic = LogisticRegression(C=1.0)
logic.fit(x_train, y_train)
print("权重：", logic.coef_)
print("预测准确率：", logic.score(x_test, y_test))
y_predict = logic.predict(x_test)
print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))
"""
召回率：           precision    recall  f1-score   support

          良性       0.98      1.00      0.99       113
          恶性       1.00      0.97      0.98        58

    accuracy                           0.99       171
   macro avg       0.99      0.98      0.99       171
weighted avg       0.99      0.99      0.99       171

有三个癌症病人没有预测出来！
"""