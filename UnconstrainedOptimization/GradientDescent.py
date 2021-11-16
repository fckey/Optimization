import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
演示最速下降法解决一个凸函数的例子：
解决问题：
    f(x,y)  = x**2 + 2y**2 - 2xy - 2
"""

def cvx_function_gradient(x, y):
    """
    计算函数的梯度
    :param x:
    :param y:
    :return:
    """
    x_grad = 2 * x - 2 * y
    y_grad = 4 * y - 2 * x
    return x_grad, y_grad


def cvx_function(x, y):
    """
    函数值的计算
    :param x:
    :param y:
    :return:
    """
    z1 = x**2
    z2 = y**2
    return z1 + 2 * z2 - 2 * x * y - 2

def generate_points(x_start, y_start, f, grad, alpha, steps):
    """
    根据优化的过程生成过程的点列
    :param x_start: 开始节点
    :param y_start: 开始节点
    :param f: 函数
    :param grad: 计算梯度的函数
    :param alpha:学习率
    :param steps:迭代的函数
    :return: 优化点列中各个x，y以及对应的函数值
    """
    X, Y, Z = [x_start], [y_start], [f(x_start, y_start)]
    for i in range(1, steps):
        # 上一个点的梯度
        grad_x, grad_y = grad(X[i - 1], Y[i - 1])
        # 按照梯度的方向移动
        new_x, new_y = X[i - 1] - alpha * grad_x, Y[i - 1] - alpha * grad_y
        # 将点放入到点的集合中
        Z.append(f(new_x, new_y))
        X.append(new_x)
        Y.append(new_y)
    return X, Y, Z

def generate_grid(x_1, x_2, y_1, y_2, delta, f):
    """
    生成一个二维网格，并计算网格中的各个点，用于后续的2d或者是3d的绘图
    :param x_1: 二维网格的坐标轴最小值
    :param x_2: 二维网格x的最大值
    :param y_1: 二维网格y的最小值
    :param y_2: 二维网格y的最大值
    :param delta: 网格中各点的分散间隔
    :param f: 函数，生成的函数
    :return: 网格坐标，和网格各点的函数值
    """
    x = np.arange(x_1, x_2, delta)
    y = np.arange(y_1, y_2, delta)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    return X, Y, Z

def generate_contours(X, Y, Z, x, y, filepath):
    """
    画图等高线图
    :param X: 网格x的坐标
    :param Y: 网格y的坐标
    :param Z: 网格中各个点的函数值变化
    :param x: 优化点列x坐标的变化
    :param y: 优化点列y坐标的变化
    :param filepath:
    :return:

     plt.figure()
    plt.contourf(X, Y, Z, 40, alpha=0.75)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    # 画出点
    plt.plot(x, y, c='r')
    plt.savefig(filepath)
    plt.show()

    """
    plt.figure()
    plt.contourf(X, Y, Z, 40, alpha=0.75,cmap=plt.cm.hot)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    # 画出点
    plt.plot(x, y, c='r')
    plt.savefig(filepath)
    plt.show()


def generate_3D_figure(X, Y, Z, x, y, z, filpath):
    """
    画出3d图片
    :param X: 网格中的x坐标
    :param Y: 网格中的y坐标
    :param Z: 网格中的z坐标
    :param x: 点列中x的变化
    :param y: 点列中y的变化
    :param z: 点列中z的变化
    :param filpath:
    :return:
    """
    fig = plt.figure()
    # 加上一个3d坐标轴
    ax = Axes3D(fig)
    p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    # 画图当前点的轨迹
    ax.plot3D(x, y, z, c='r', linewidth=1)
    plt.colorbar(p, shrink=0.8)
    plt.savefig(filpath)
    plt.show()
    pass
if __name__ == "__main__":
    # 初始化网格数据
    x_1, x_2, y_1, y_2,  delta = -4.0, 4.0, -4.0, 4.0, 0.025
    # 开始梯度下降的参数初始化
    x_start, y_start = 3.6, 3.6
    # 学习率
    alpha = 0.01
    # 步长
    step = 10000
    # 返回的是原来的初始网格数据
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    X_P, Y_P, Z_P = generate_points(x_start, y_start, cvx_function, cvx_function_gradient, alpha, step)
    generate_contours(X, Y, Z, X_P, Y_P, './figures/gd2d.png')
    generate_3D_figure(X, Y, Z, X_P, Y_P, Z_P, "./figures/gd3d.png")


