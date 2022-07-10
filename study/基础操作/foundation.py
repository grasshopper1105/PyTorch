import random
import torch
import numpy as np

a = torch.randn(2, 3)  # 生成2行3列满足正态分布的张量
A = a.cuda()  # 将张量存放在GPU
b = isinstance(a, torch.FloatTensor)  # 验证a是否是某类型
c = torch.tensor(5)
d = torch.tensor([1, 1])
D = torch.Tensor(2, 3)

f = np.arange(1, 5, 0.5)
F = torch.from_numpy(f)  # 将numpy格式转为tensor

g = torch.ones(2, 2)  # 生成全一矩阵
G = torch.eye(3)  # 生成3*3对角为1其他为0的矩阵

h = torch.rand(5, 5)  # 随机生成5*5均匀分布的张量
H = torch.randint(1, 10, (3, 3))  # 随机生成[1,10)的3*3int张量
K = torch.rand_like(h)  # 读取h.shape,生成均匀分布的张量

i = h.shape[0]
I = h.size()  # 两者等价
j = h.dim()  # 几维张量
J = h.numel()  # 元素个数

# 初始化空间使用，数据未初始化
e = torch.FloatTensor(6, 6)  # 随机生成Float类型的6*6张量
E = torch.IntTensor(6, 6)  # 随机生成Int类型的6*6张量
k = torch.empty(6, 6)  # 生成空张量

torch.set_default_tensor_type(torch.DoubleTensor)  # 设置torch.Tensor和torch.tensor格式

l = torch.normal(mean=torch.full([12], 0.), std=torch.linspace(1, -1, steps=12)).reshape(3, 4)  # 生成均值为0，方差为1~-1递减的3*4张量
L = torch.logspace(1, -1, steps=10)  # 生成10个数，从10^1 ~ 10^-1

m = torch.take(h, torch.tensor([0, 2, 4]))  # 将张量打平后取出指定位置的元素
n = torch.randn(4, 3)
N = n.t()  # 转置，只能针对2D
o = n.prod()  # 累乘

# 　打乱数据
torch.random.seed()
rand1 = torch.rand(2, 3)
rand2 = torch.rand(2, 2)
idx = torch.randperm(2)  # 打乱序列
rand1 = rand1[idx]
rand2 = rand2[idx]  # 保证数据配对


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


# 设置随机数种子
setup_seed(20)

# 选取指定目标
goal = torch.randn(3, 3, 3)
goal1 = goal.index_select(dim=1, index=torch.tensor([1, 2]))  # 取出第一个维度的第一和第二个位置的内容

# ...
pdd = torch.rand(4, 3, 28, 28)
pdd1 = pdd[...]  # 取出全部维度
pdd2 = pdd[:2, ..., ::2]  # [2,3,28,14]
pdd3 = pdd[0, ...]  # [3,28,28]

# mask_selected
lbw = torch.randn(3, 4)
lbw1 = lbw.ge(0.5)  # 将所有>=0.5的数变为1，其余为0
lbw2 = torch.masked_select(lbw, lbw1)  # 取出所有为1的位置的数，组成一维向量

# squeeze & unsqueeze
nb = torch.rand(1, 2, 3, 4)
nb1 = nb.unsqueeze(0)  # 在第0个位置插入维度 [1,1,2,3,4]
nb2 = nb.unsqueeze(-2)  # 在倒数第二个位置插入维度 [1,2,3,1,4]
nb3 = nb.unsqueeze(0).unsqueeze(1).unsqueeze(2)
nb4 = nb.squeeze(1)  # 挤压掉第一个位置 ps:只有该维度元素个数为1时才能删除
nb5 = nb.squeeze()  # 挤压所有元素个数为1的维度

# expand & repeat
bn = torch.rand(1, 2, 1)
bn1 = bn.expand(3, 2, 3)  # 将原先元素为1的维度扩展为指定大小
bn2 = bn.expand(-1, 2, -1)  # -1时该维度保持不变
bn3 = bn.repeat(3, 2, 3)  # 将每个维度复制n次 [3，4，3] # 不推荐repeat

# transpose & permute
emp = torch.randn(4, 3, 32, 32)
emp1 = emp.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).contiguous().transpose(1, 3)
# transpose交换指定两个维度，contiguous重新申请内存让数据连续，view = reshape
res = torch.all(torch.tensor(torch.equal(emp, emp1)))  # all:所有元素为1时返回True；equal:两个张量相同时返回True
emp2 = emp.permute(0, 2, 3, 1)  # 将原来维度的元素放在指定位置 [4,32,32,3]

# cat & stack
abc1 = torch.rand(2, 3, 4)
abc2 = torch.rand(3, 3, 4)
abc3 = torch.rand(2, 3, 4)
abc = torch.cat([abc1, abc2], dim=0)  # 在指定维度拼接，变成[5,3,4]
ABC = torch.stack([abc1, abc3], dim=0)  # 在指定维度上堆叠，只有shape相同才能使用，变成[2,2,3,4]

# split & chunk
aa = torch.rand(32, 16, 8)
bb1, bb2, bb3 = aa.split([5, 5, 6], dim=1)  # 将aa按照第一维拆成长度为[5,5,6]的张量
cc1, cc2 = aa.split(4, dim=2)  # 将aa按照第二维拆成长度全为2的张量
dd1, dd2 = aa.chunk(2, dim=0)  # 将aa按照第0维拆成两个张量

# * & matmul
aaa = torch.rand(4, 3, 28, 64)
bbb = torch.rand(4, 1, 64, 32)
# ccc1 = aaa * bbb # 对应元素相乘
ccc2 = torch.matmul(aaa, bbb)  # 矩阵乘法 等价于 aaa @ bbb 超过两维时只对后两维计算，变成[4,3,28,32]

# clamp裁剪
aaaa = torch.rand(3, 3) * 15
aaaa1 = aaaa.clamp(10)  # 将小于10的数都变成10
aaaa2 = aaaa.clamp(0, 10)  # 将(0,10)之外的数都变成10

# norm lp范数
bbbb = torch.rand(1, 2, 3)
bbbb1 = bbbb.norm(2)  # l2范数
bbbb2 = bbbb.norm(1, dim=1)  # 在第一维求l1范数，得到[1,3]张量

# topk & kthvalue
cccc = torch.rand(4, 10) * 15
cccc1 = cccc.topk(2, dim=0, largest=True)  # 返回第0个维度上前2大的数据 #largest=False时返回最小的k个
cccc2 = cccc.kthvalue(3, dim=0)  # 返回第0个维度上第3小的数据

#　where(cond,x,y)
cond = torch.rand(2, 2)
x = torch.zeros(2, 2)
y = torch.ones(2, 2)
result = torch.where(cond > 0.5, x, y)  # cond大于0.5的位置放x对应位置的元素，否则放y的
print(cond, '\n', x, '\n', y, '\n', result)

# gather 查表
prob = torch.randn(4, 10)
idx = prob.topk(3, dim=1)[1]  # 取出第一维前三大的数的索引
label = torch.arange(10) + 100  # 生成源表
Result = torch.gather(label.expand(4, 10), dim=1, index=idx.long())  # 在第一个维度上映射 ps: 1->101 9->109
