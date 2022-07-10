import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

'''初始化'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 0.01
momentum = 0.5
num_epochs = 2

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''准备训练集(train=True)'''
# transforms.ToTensor将图片转化tensor, Normalize((mean,), (std,))为标准化
train_data = datasets.MNIST('./mnist', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))]))
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)

'''准备测试集(train=False)'''
test_data = datasets.MNIST('./mnist', train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))]))
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=2, pin_memory=True)

'''Normalize中计算所有图片的mean和std'''
# data = [d[0].data.cpu().numpy() for d in mnist]
# np.mean(data) 0.1307
# np.std(data) 0.3081

'''搭建CNN'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        # Conv2d(输入通道，输出通道，卷积核大小(5,5)，步长为(1,1)，不用0填充，1为周围一圈0,2为两圈)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)  # Linear(输入维度，中间维度)
        self.fc2 = nn.Linear(500, 10)  # Linear(中间维度，输出维度)

    '''output_shape = (input_shape - filter_shape + 2*padding) / stride + 1'''

    def forward(self, x):
        # 输入x维度 (1,28,28)
        x = F.relu(self.conv1(x))
        # 28+1-5 -> (20,24,24)
        x = F.max_pool2d(x, 2, 2)  # max_pool2d(输入，卷积核(2,2)，步长(2,2))
        # 24/2 -> (20,12,12)
        x = F.relu(self.conv2(x))
        # 12+1-5 -> (50,8,8)
        x = F.max_pool2d(x, 2, 2)
        # 8/2 -> (50,4,4)
        x = x.view(-1, 4 * 4 * 50)
        # (50,4,4) -> (1,4*4*50)
        x = F.relu(self.fc1(x))
        # (1,4*4*50) -> (1,500)
        x = self.fc2(x)
        # (1,500) -> (1,10)
        return F.log_softmax(x, dim=1)


'''训练数据'''


def train(model, device, train_dataloader, optimizer, epoch):
    model.train()  # 训练模式 BN有区别
    for idx, (data, target) in enumerate(train_dataloader):  # enumerate枚举容器内元素
        data, target = data.to(device), target.to(device)
        pred = model(data)  # pred:(batch_size,10)
        loss = F.nll_loss(pred, target)  # CE

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('Train Epoch:{}, iter: {:>d}, Loss: {:>.8f}'.format(
                epoch, idx, loss.item()))


'''测试数据'''


def test(model, device, test_dataloader):
    model.eval()  # 测试模式
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):  # enumerate枚举容器内元素
            data, target = data.to(device), target.to(device)
            output = model(data)  # 找出10个预测值中最大的位置
            total_loss += F.nll_loss(output, target, reduction='sum').item()  # CE reduction默认为'mean',即求loss平均
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(target)).sum().item()

    total_loss /= len(test_dataloader.dataset)
    acc = correct / len(test_dataloader.dataset)
    print('----------------------------------------')
    print('Test Loss: {:>.5f}, Accuracy: {:>%}'.format(
        total_loss, acc))
    print('----------------------------------------')


'''神经网络'''
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def main():
    for epoch in range(num_epochs):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)
    torch.save(model.state_dict(), 'mnist_cnn.pt')


if __name__ == "__main__":
    main()
