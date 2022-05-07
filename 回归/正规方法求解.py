import numpy as np

data = np.loadtxt('../dataset/food_truck_data.txt', delimiter=",") # data是m*2矩阵，每一行表示一个样本
train_x = data[:, 0]    # 城市人口，  m*1矩阵
train_y = data[:, 1]    # 餐车利润，  m*1矩阵

X = np.ones(shape=(len(train_x), 2))
X[:, 1] = train_x
y = train_y


XT = X.transpose()  # 求转置

XTy = XT @ y

w = np.linalg.inv(XT@X) @ XTy
print(w)    # [-3.89578088  1.19303364]

# 预测人口4.6（万）城市的餐车利润：
print(4.6*w[1]+w[0])    # 1.5921738849602658