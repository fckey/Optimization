"""
演示拟牛顿法中的BFGS解决解决凸函数优化的问题：
    需要待解决的问题是：
     f(x,y)  = x**2 + 2y**2 - 2xy - 2
"""
import matplotlib.pyplot as plt
import numpy as np
from  mpl_toolkits.mplot3d import Axes3D

def cvx_function(X):
    """
     凸函数函数值的计算
    :param X:
    :return:
    """
    z1 = X[0]**2
    z2 = X[1]**2
    return z1 + 2 * z2 - 2 * X[0] * X[1] - 2

def cvx_function_gradient(X):
    """
    计算函数的梯度
    :param x:
    :return:
    """
    grad_x = 2 * X[0] - 2 * X[1]
    grad_y = 4 * X[1] - 2 * X[0]
    return np.array([grad_x, grad_y])

def generate_grid(x_1, x_2, y_1, y_2, delta, f):
    """
    计算网格中生成的各个点
    :param x_1: x轴开始的最小点
    :param x_2: x轴结束的最后一个点
    :param y_1: y轴开始的一个点
    :param y_2: y轴结束的一个点
    :param delta: 间隔
    :param f: 函数值
    :return:
    """
    x = np.arange(x_1, x_2, delta)
    y = np.arange(y_1, y_2, delta)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    return X, Y, Z

def plot_2D_figure(X, Y, Z, x, y, filepath):
    """
    画二维图
    """
    plt.figure()
    plt.contourf(X, Y, Z, 10,cmap=plt.cm.hot)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    plt.plot(x, y, c='r')
    plt.savefig(filepath)
    plt.show()


def plot_3D_figure(X, Y, Z, x, y, z, filepath):
    """
    画三维图
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x, y, z, c='r', linewidth=2)
    plt.colorbar(p, shrink=0.8)
    plt.savefig(filepath)
    plt.show()

def generate_points(x_start, f, grad, epsilon=1e-5, steps=100):
    """
    根据拟牛顿法（DFP）生成优化点列的过程
    :param x_start: 起始点的坐标
    :param f: 需要优化的函数
    :param grad: 计算f函数的梯度函数
    :param epsilon: 迭代停止的条件，当当前点的梯度的模小于epsilon时，迭代停止
    :param steps: 最大的迭代步数
    :return: 优化过程生成点列的x坐标序列，y坐标序列，以及每一个点对应的函数值
    """
    # 开始的位置
    X, x_old = x_start, x_start
    # 计算当前的值
    Z = f(x_start)
    # 开始初始化b矩阵
    B_old = np.mat(np.eye(2, dtype=np.float32) * 0.01)
    grad_old = np.mat(grad(x_old))
    I = np.mat(np.eye(2, dtype=np.float32))
    for i in range(1, steps):
        # 计算一个新的位置
        x_new = x_old - np.array(B_old * grad_old)
        grad_new = np.mat(grad(np.array(x_new)))
        # 判断当前是否满足条件
        if np.sqrt(np.sum(grad_new.T * grad_new)) < epsilon:
            # 添加x当前位置
            X = np.concatenate((X, x_new), axis=1)
            # 获取初始值
            z_new = f(np.array(x_new))
            Z = np.concatenate((Z, z_new))
            print("最后所花的步数: ", i)
            print("最后一次x,y的变量值: ", [x_new[0], x_new[1]])
            print("最终的优化函数值:", z_new)
            break
        y = np.mat(grad_new - grad_old)
        s = np.mat(x_new - x_old)
        # 更新当前的值
        B_new = (I - (s * y.T) / (s.T * y)) * B_old * (I - (s * y.T) / (s.T * y)) + (s * s.T) / (s.T * y)
        X = np.concatenate((X, x_new), axis=1)
        z_new = f(np.array(x_new))
        Z = np.concatenate((Z, z_new))
        B_old = B_new
        grad_old = grad_new
        x_old = x_new
    return X[0], X[1], Z


if __name__ == "__main__":
    x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    x_start = np.array([[3.6], [3.6]])
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    x, y, z = generate_points(x_start, cvx_function, cvx_function_gradient)
    plot_2D_figure(X, Y, Z, x, y, './figures/bfgs2d.png')
    plot_3D_figure(X, Y, Z, x, y, z, './figures/bfgs3d.png')