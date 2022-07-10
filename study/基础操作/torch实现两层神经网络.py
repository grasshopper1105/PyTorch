import torch

'''初始化'''
# N:64个输入的训练数据 D_in:输入1000维 H:隐藏层100层 D_out:输出10维
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).to('cuda:0')  # 输入数据
y = torch.randn(N, D_out).to('cuda:0')  # 输出数据
w1 = torch.randn(D_in, H).to('cuda:0')  # h = w1 * x
w2 = torch.randn(H, D_out).to('cuda:0')  # y_hat = w2 * relu(h)
learning_rate = 1e-6

'''神经网络'''
for it in range(500):
    '''前向传播(forward pass)'''
    h = x.mm(w1)  # (N, H)
    h_relu = h.clamp(min=0)  # (N, H)
    y_pred = h_relu.mm(w2)  # (N, D_out)

    '''loss'''
    loss = (y_pred - y).pow(2).sum().item()  # mse
    print(it, loss)

    '''后向传播(backward pass)'''
    # 梯度下降,反向求导
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 更新w1，w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
