import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

def gradient_descent(x, y, w, b, alpha=0.01, iterations=100, epsilon=1e-9):
    _history = []
    for i in range(iterations):
        dw = np.mean((w * x + b - y) * x)
        db = np.mean((w * x + b - y))
        if abs(dw) < epsilon and abs(db) < epsilon:
            break;

        # 更新w: w = w - alpha * gradient
        w -= alpha * dw
        b -= alpha * db
        _history.append([w, b])

    return _history


# 绘制模型参数对应的假设函数的图像
def draw_line(plt,w,b,x,linewidth =2):
    m=len(x)
    f = [0]*m
    for i in range(m):
       f[i] = b+w*x[i]
    plt.plot(x, f, linewidth)


def loss(x,y,w,b):
    m = len(y)
    return np.mean((x*w+b-y)**2)/2

#   绘制损失函数对应的曲面
def plot_history(x, y, history, figsize=(20, 10)):
    w = [e[0] for e in history]
    b = [e[1] for e in history]

    xmin, xmax, xstep = min(w) - 0.2, max(w) + 0.2, .2
    ymin, ymax, ystep = min(b) - 0.2, max(b) + 0.2, .2
    ws, bs = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))

    zs = np.array([loss(x, y, w, b) for w, b in zip(np.ravel(ws), np.ravel(bs))])
    z = zs.reshape(ws.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('w[0]', labelpad=30, fontsize=24, fontweight='bold')
    ax.set_ylabel('w[1]', labelpad=30, fontsize=24, fontweight='bold')
    ax.set_zlabel('L(w,b)', labelpad=30, fontsize=24, fontweight='bold')

    ax.plot_surface(ws, bs, z, rstride=1, cstride=1, color='b', alpha=0.2)

    w_sart, b_start, w_end, b_end = history[0][0], history[0][1], history[-1][0], history[-1][1]
    ax.plot([w_sart], [b_start], [loss(x, y, w_sart, b_start)], markerfacecolor='b', markeredgecolor='b', marker='o',
            markersize=7)
    ax.plot([w_end], [b_end], [loss(x, y, w_end, b_end)], markerfacecolor='r', markeredgecolor='r', marker='o',
            markersize=7)

    z2 = [loss(x, y, w, b) for w, b in history]
    ax.plot(w, b, z2, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)
    ax.plot(w, b, 0, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)

    fig.suptitle("L(w,b)", fontsize=24, fontweight='bold')
    return ws, bs, z


data = np.loadtxt('../dataset/food_truck_data.txt', delimiter=",") # data是m*2矩阵，每一行表示一个样本
train_x = data[:, 0]    # 城市人口，  m*1矩阵
train_y = data[:, 1]    # 餐车利润，  m*1矩阵

X = train_x
y = train_y

w,b = 0.,0.

alpha = 0.02
iterations=1000
history = gradient_descent(X,y,w,b,alpha,iterations)
print(len(history)) # 1000
print(history[-1]) # [1.1822480052540145, -3.7884192615511796]
print('---------------')

'''

fig, ax = plt.subplots()
plt.scatter(X, y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10,000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10,000s", fontsize=14)
plt.axis([4, 25, -5, 25])
w,b = history[-1]
draw_line(plt,w,b,X,6)
plt.show()

#   绘制损失曲线
costs = [loss(X,y,w,b) for w,b in history]
plt.axis([0, len(costs), 4, 6])
plt.plot(costs)
plt.show()


ws,bs,z = plot_history(X,y,history)

# 对每个学习率绘制对应的代价历史曲线
plt.figure()
num_iters = 1200
learning_rates = [0.01, 0.015, 0.02]
for lr in learning_rates:
    w,b=0,0
    history = gradient_descent(X, y,w, b,lr, num_iters)
    cost_history = [loss(X,y,w,b) for w,b in history]
    plt.plot(cost_history, linewidth=2)
plt.title("Gradient descent with different learning rates", fontsize=16)
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.legend(list(map(str, learning_rates)))
plt.axis([0, num_iters, 4, 6])
plt.grid()
plt.show()

'''
#   梯度验证
df_approx = lambda x,y,w,b,eps: ( (loss(x,y,w+eps,b)-loss(x,y,w-eps,b) )/(2*eps)
                                ,  (loss(x,y,w,b+eps)-loss(x,y,w,b-eps) )/(2*eps) )
#   在任意一个点如 (𝑤,𝑏)=(1.0,−2.0) 比较分析和数值梯度。
w =1.0
b = -2.
eps = 1e-8
dw = np.mean((w*X+b-y)*X)
db = np.mean((w*X+b-y))
grad = np.array([dw,db])
grad_approx = df_approx(X,y,w,b,eps)
print(grad)
print(grad_approx)
print(abs(grad-grad_approx))

#用求得的w求X种样本的预测值
m=len(X)
predictions = [0]*m
for i in range(m):
    predictions[i] =  X[i]*w+b

plt.scatter(X, y, marker="x", c="red")
plt.scatter(X, predictions, marker="o", c="blue")
#plt.plot(X, predictions, linewidth=2)  # plot the hypothesis on top of the training data
plt.show()