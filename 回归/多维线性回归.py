import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n_points = 20
a = 3
b = 2
c  = 5

x_range = 5
y_range = 5
noise = 3

xs = np.random.uniform(-x_range,x_range,n_points)
ys = np.random.uniform(-y_range,y_range,n_points)
zs = xs*a+ys*b+ c+ np.random.normal(scale=noise)

# xs[:, None]转换位列向量
X = np.hstack((xs[:, None],ys[:, None]))
y = zs

def linear_regression_vec(X, y, alpha, num_iters,gamma = 0.8,epsilon=1e-8):
    history = []          # 记录迭代过程中的参数
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype),X))  #添加一列特征1
    num_features = X.shape[1]
    v= np.zeros_like(num_features)
    w = np.zeros(num_features)
    for n in range(num_iters):
        predictions = X @ w                 #求假设函数的预测值，即f(x)
        errors = predictions - y                # 预测值和真实值的误差
        gradient = X.transpose() @ errors /len(y)        #计算梯度
        if np.max(np.abs(gradient))<epsilon:
            print("gradient is small enough!")
            print("iterated num is :",n)
            break
        #w -= alpha * gradient        #更新模型的参数
        v = gamma*v+alpha* gradient
        w= w-v
        history.append(w)
        #cost_history.append((errors**2).mean()/2)     # compute and record the cost
    return history                  # return the history of optimized parameters


def compute_loss_history(X,y,w_history):
    loss_history = []
    for w in w_history:
        errors = X@w[1:]+w[0]-y
        loss_history.append((errors**2).mean()/2)
    return loss_history


learning_rate = 0.02
num_iters = 100
history = linear_regression_vec(X, y,learning_rate, num_iters)
print("w:",history[-1])

loss_history = compute_loss_history(X,y,history)
print(loss_history[:-1:10])
plt.plot(loss_history, linewidth=2)
plt.title("Gradient descent with learning rate = " + str(learning_rate), fontsize=16)
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.grid()
plt.show()