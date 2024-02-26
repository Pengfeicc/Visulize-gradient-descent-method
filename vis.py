import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from models import QuadraticModel
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# 设置全局字体和大小
mpl.rc('font', family='Arial', size=14)
# 创建一个从浅绿(#C5DE89)到深绿(#658D67)的渐变色图
cmap = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#658D67","#C5DE89"])

def true_func_wraper(x1, x2, w, sample=False):
    return w[0] * x1 ** 2 + w[1] * x2 ** 2

def true_func(xs, ws, sample=False):
    res = np.dot(xs ** 2, ws)
    if sample:
        res = res + np.random.normal(loc=0, scale=10, size=res.shape)
    return res

def plot_3d(x, y, z, style='scatter', ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')

    if style == 'surface':
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cmap,
                        linewidth=0, antialiased=False)
    elif style == 'scatter':
        ax.scatter(x, y, z, edgecolor='none', marker='.')
    else:
        raise NotImplementedError

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    return ax
ref_w = [10, 4]

ref_xs = np.array([np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)]).T
ref_ys = true_func(ref_xs, ref_w)

X1, X2 = np.meshgrid(ref_xs[:,0], ref_xs[:,1])
Y = true_func_wraper(X1, X2, ref_w)

xs_samp = np.random.randint(-1000, 1000, size=(1000, 2)) / 1000
x1s_samp = xs_samp[:,0]
x2s_samp = xs_samp[:,1]
ys_samp = true_func_wraper(x1s_samp, x2s_samp, ref_w, sample=True)
ax = plot_3d(X1, X2, Y, 'surface')
plot_3d(x1s_samp, x2s_samp, ys_samp, style='scatter', ax=ax)


models = []
methods = [
    'bgd',
    'sgd',
    'mbgd',
]
n_epochs = [500] + [2] * (len(methods) - 1)

for met, epo in zip(methods, n_epochs):
    # 使用met而不是method
    if met == 'mbgd':
        # 为mbgd指定batch_size参数
        model = QuadraticModel(learning_rate=0.1, n_epochs=epo)
        model.fit(xs_samp, ys_samp, method=met, batch_size=5)
    else:
        # 对于bgd和sgd，不需要batch_size参数
        model = QuadraticModel(learning_rate=0.1, n_epochs=epo)
        model.fit(xs_samp, ys_samp, method=met)
    models.append(model)

# 假设 models 和 methods 已经定义并包括了 bgd, sgd, 以及 mbgd
fig, axes = plt.subplots(1, len(models), figsize=(18, 6), sharex=True, sharey=True)  # 适应模型数量动态调整subplot数量
axes = axes.ravel()  # 平铺axes以便循环访问

for k, (model, method) in enumerate(zip(models, methods)):
    epochs = range(len(model.history['loss']))  # epoch数
    loss_values = model.history['loss']  # 损失值

    ax = axes[k]
    ax.plot(epochs, loss_values, '-o', label=f"{method}", color='#97C139', linewidth=1, markersize=2)
    ax.set_title(f"{method}, epochs: {len(epochs)}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

plt.tight_layout()

#line version
# 绘图
w1_samp = np.arange(-ref_w[0] - 0, ref_w[0] + 5, 1)
w2_samp = np.arange(-ref_w[1] - 3, ref_w[1] + 10, 1)
W1, W2 = np.meshgrid(w1_samp, w2_samp)
J = np.zeros(W1.shape)

# 计算损失
for i, _w1 in enumerate(w1_samp):
    for j, _w2 in enumerate(w2_samp):
        J[j, i] = np.mean((true_func_wraper(x1s_samp, x2s_samp, [_w1, _w2]) - ys_samp) ** 2)

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制损失的登高线图
ctf = ax.contourf(W1, W2, J, 20, cmap=cmap)
fig.colorbar(ctf, ax=ax)
ax.scatter(*ref_w, marker='*', color='orange', s=100, label='Reference')

# 绘制三种方法的轨迹
colors = ['red', 'yellow', 'orange']
labels = ['BGD', 'SGD', 'MBGD']

for model, color, label in zip(models, colors, labels):
    ax.plot(model.history['w'][:, 0], model.history['w'][:, 1], marker='.', color=color, lw=0.5, markersize=5, label=label)

ax.set_title('Gradient Descent Methods Comparison')
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.legend()

plt.show()