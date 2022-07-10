import numpy as np

'''初始化'''
# N:64个输入的训练数据 D_in:输入1000维 H:隐藏层100层 D_out:输出10维
N, D_in, H, D_out = 64, 1000, 100, 10
x = np.random.randn(N, D_in)  # 输入数据
y = np.random.randn(N, D_out)  # 输出数据
w1 = np.random.randn(D_in, H)  # h = w1 * x
w2 = np.random.randn(H, D_out)  # y_hat = w2 * relu(h)
learning_rate = 1e-6

'''神经网络'''
for it in range(500):
    '''前向传播(forward pass)'''
    h = x.dot(w1)  # (N, H)
    h_relu = np.maximum(h, 0)  # (N, H)
    y_pred = h_relu.dot(w2)  # (N, D_out)

    '''loss'''
    loss = np.square(y_pred - y).sum()  # mse
    print(it, loss)

    '''后向传播(backward pass)'''
    # 梯度下降,反向求导
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新w1，w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
