import torch
import numpy as np

'''矩阵生成'''
a1 = torch.empty(5, 3)  # 随机生成未初始化的空矩阵
a2 = torch.zeros(5, 3)  # 生成全0矩阵
a3 = torch.rand(5, 3)  # 生成[0,1]之间的随机数
a4 = torch.randn(5, 3)  # 生成符合标准正态分布N(0,1)的随机数
a5 = torch.tensor([5, 3.3])  # 给定数据生成矩阵
a6 = a5.new_ones((5, 3))  # 保留a5的数据类型等特征
a7 = torch.randn_like(a6, dtype=torch.float)  # 生成和a6形状相同的符合标准正态分布N(0,1)的随机数
# print(a7)

'''类型转换'''
b1 = torch.zeros(5, 3, dtype=torch.long)  # 长整形
b2 = torch.zeros(5, 3).long()  # 长整形

'''得到形状'''
# print(a7.shape)
# print(a7.size())

'''矩阵运算'''
x = torch.randn(5, 3)
y = torch.rand(5, 3)
result1 = x + y  # 简化版
result2 = torch.empty(5, 3)
torch.add(x, y, out=result2)  # 复杂版
z = y.add_(x)  # 带_的操作会改变y的值,即此时y，z相同

'''Indexing'''
c = torch.randn(5, 3)
c1 = c[1:, 1:]  # 与numpy相同

'''Resize'''
d = torch.randn(2, 8)
d1 = d.view(16)  # 展平为16维，同reshape
d2 = d.view(-1, 2)
d3 = d.transpose(1, 0)  # 交换第1维和第0维
# print(d3)

'''取出仅有一个元素的矩阵中的元素值'''
e = torch.randn(1)
e1 = e.item()
# print(e1)

'''NumPy和torch相互转换'''
f1 = torch.ones(5)
f2 = np.ones(5)
f3 = f1.numpy()  # torch到NumPy
f4 = torch.from_numpy(f2)  # NumPy到torch

''' CUDA Tensors
    numpy的所有操作都在CPU上，torch可以转至GPU'''
if torch.cuda.is_available():
    device = torch.device('cuda')
    # 转移至GPU的两种方式
    g1 = torch.rand(5, 3, device=device)
    g2 = torch.rand(5, 3).to(device)
    # 转移至CPU的两种方式
    g1.to('cpu')
    g2.cpu()

    model = torch.ones(5,3)
    model = model.cuda()  # 将模型转移至cuda
