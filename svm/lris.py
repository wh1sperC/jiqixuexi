import numpy as np
from sklearn import svm

# 导入数据
x = np.loadtxt("./svm/Iris/train/x.txt")
y = np.loadtxt("./svm/Iris/train/y.txt")
x1 = np.loadtxt("./svm/Iris/test/x.txt")
y1 = np.loadtxt("./svm/Iris/test/y.txt")

# 训练svm分类器
classifiers = svm.SVC(C=0.2, kernel='rbf', gamma=100, decision_function_shape='ovo')  # 各项参数
classifiers.fit(x, y.ravel())

# 打印结果
print("训练集：", classifiers.score(x, y))
print("测试集: ", classifiers.score(x1, y1))
print('train_decision_function:\n', classifiers.decision_function(x))
print('predict_result:\n', classifiers.predict(x))
