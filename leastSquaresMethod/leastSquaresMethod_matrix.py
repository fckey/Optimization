import numpy as np
m = int(input("input m "))
n = int(input("input n"))
# 假设真正的平面是
true_w = np.random.randint(1,10,size=n)
true_b = np.random.rand()
print(true_w)
print(true_b)

def data(w, b, m):
    """生成 y = xw + b + 噪音"""
    #  normal 均值为0  方差是1， 需要n个样本
    X = np.random.normal(0, 1, (m, len(w)))
    Y = np.matmul(X, w) + b
    Y += np.random.normal(0, 0.01, Y.shape)
    return X, Y.reshape((-1, 1))
def matmul(X,Y):
    return np.matmul(X,Y)

data = data(true_w, true_b, m)
X = np.array(data[0])
print(X.T)
Y = np.array(data[1])
print(Y.T)
B = matmul(matmul(np.matrix(matmul(X.T,X)).I,X.T),Y)
print("=" * 10)
print(B.T)
