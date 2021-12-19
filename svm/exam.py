import numpy as np
from sklearn import svm

# 导入数据
X = np.loadtxt("./svm/Exam/train/x.txt")
Y = np.loadtxt("./svm/Exam/train/y.txt")
X1 = np.loadtxt("./svm/Exam/test/x.txt")
Y1 = np.loadtxt("./svm/Exam/test/y.txt")

# 训练svm分类器
classifier = svm.SVC(C=2, kernel='linear', gamma=100, decision_function_shape='ovr')  # 各项参数
classifier.fit(X, Y.ravel())

# 打印结果
print("训练集：", classifier.score(X, Y))
print("测试集: ", classifier.score(X1, Y1))
print('train_decision_function:\n', classifier.decision_function(X))
print('predict_result:\n', classifier.predict(X))
