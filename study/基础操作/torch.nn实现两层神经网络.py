import torch
import torch.nn as nn

'''初始化'''
# N:64个输入的训练数据 D_in:输入1000维 H:隐藏层100层 D_out:输出10维
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)  # 输入数据
y = torch.randn(N, D_out)  # 输出数据

'''定义模型'''
model = torch.nn.Sequential(
    nn.Linear(D_in, H),  # h = w_1 * x + b_1
    nn.ReLU(),  # a = relu(h)
    nn.Linear(H, D_out)  # y = w_2 * a + b_2
)

# '''初始化权重'''
# nn.init.normal_(model[0].weight)
# nn.init.normal_(model[2].weight)
# model = model.cuda()

'''神经网络'''
loss_fn = nn.MSELoss(reduction='sum')  # loss
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
for it in range(500):
    '''前向传播(forward pass)'''
    y_pred = model(x)

    '''loss'''
    loss = loss_fn(y_pred, y)
    print(it, loss.item())

    optimizer.zero_grad()
    '''后向传播(backward pass)'''
    loss.backward()

    '''更新参数'''
    optimizer.step()
