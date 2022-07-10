import torch

'''初始化'''
# N:64个输入的训练数据 D_in:输入1000维 H:隐藏层100层 D_out:输出10维
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)  # 输入数据
y = torch.randn(N, D_out)  # 输出数据
w1 = torch.randn(D_in, H, requires_grad=True)  # h = w1 * x
w2 = torch.randn(H, D_out, requires_grad=True)  # y_hat = w2 * relu(h)
learning_rate = 1e-6

'''神经网络'''
for it in range(500):
    '''前向传播(forward pass)'''
    y_pred = x.mm(w1).clamp(min=0).mm(w2)  # clamp:保留[min，max]之间值，其余变成min或max

    '''loss'''
    loss = (y_pred - y).pow(2).sum()  # mse
    print(it, loss.item())

    '''后向传播(backward pass)'''
    loss.backward()

    # 更新w1，w2,只记住正向梯度
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()  # 梯度清零
        w2.grad.zero_()  # 梯度清零
