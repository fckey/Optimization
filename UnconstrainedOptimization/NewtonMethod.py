"""
演示牛顿法解决凸函数优化的问题：
    需要待解决的问题是：
     f(x,y)  = x**2 + 2y**2 - 2xy - 2

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cvx_function(X):
    """
    凸函数函数值的计算
    """
    z1 = X[0]**2
    z2 = X[1]**2
    z = z1 + 2 * z2 - 2 * X[0] * X[1] - 2
    return z

def cvx_function_gradient(X):
    """
    计算函数值的梯度
    :param X:  其中向量的第一个值是x,第二个值是y
    :return:
    """
    grad_x = 2 * X[0] - 2 * X[1]
    grad_y = 4 * X[1] - 2 * X[0]
    return np.array([grad_x, grad_y])

def cvx_hessian_inverse(X):
    """
    计算该函数的hessian矩阵，牛顿法更新的方法需要其与梯度的乘积
    :return:
    """
    hessian_matrix = np.mat([[2, -2], [-2, 4]])
    hessian_inverse =np.linalg.pinv(hessian_matrix)
    return np.array(hessian_inverse)

def generate_grid(x_1, x_2, y_1, y_2, delta, f):
    """
    生成二维网格，并计算网格中各个点的值，用于后续画登高线三维图
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
    plt.contourf(X, Y, Z, 40,cmap=plt.cm.hot)
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

def generate_points(x_start, f, grad, hessian_inverse, eps=1e-10, steps = 2000):
    """
    根据牛顿法生成优化点列的过程，
    :param x_start: 起始点的坐标
    :param f: 需要被优化的函数
    :param grad:计算f函数的梯度
    :param hessian_inverse:计算f函数hessian逆矩阵的函数
    :param eps: 迭代的停止条件
    :param steps: 最大的迭代步数
    :return:
    """
    # 接受起始坐标位置
    X = x_start
    # 计算其实的位置大小
    Z = f(x_start)
    # 将当前的值进行输出
    print(Z)
    for i in range(1, steps):
        current_grad = grad(X[:, i - 1])
        # 判断当前是否到了最小值点
        if np.sqrt(np.sum(current_grad**2)) < eps:
            print("一共所花的步数: ", i)
            break
        #  求出当前hessian的reverse的形式
        current_hessian_inverse = hessian_inverse(X[:, i - 1])
        #  x移动的距离是 x - （hessian矩阵的逆 乘上一个梯度）
        x_new = X[:, i - 1].reshape(2, 1) - np.dot(current_hessian_inverse, current_grad)
        # 更新z
        z_new = f(x_new)
        print(z_new)
        # 直接在一维的时候进行拼接
        X = np.concatenate((X, x_new), axis=1)
        # 对z进行正常的拼接工作
        Z = np.concatenate((Z, z_new))
    return X[0], X[1], Z

if __name__ == "__main__":
    x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    x_start = np.array([[3.6], [3.6]])
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    x, y, z = generate_points(x_start, cvx_function, cvx_function_gradient, cvx_hessian_inverse)
    plot_2D_figure(X, Y, Z, x, y, './figures/nc2d.png')
    plot_3D_figure(X, Y, Z, x, y, z, './figures/nc3d.png')
