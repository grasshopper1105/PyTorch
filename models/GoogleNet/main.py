import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''初始化'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-4
epochs = 20

'''载入数据'''
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(45),
                                transforms.Resize([32, 32]),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

train_data = datasets.CIFAR10(root='./cifar', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='./cifar', train=False, download=False, transform=transform)

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


class Inception(nn.Module):
    """定义模型"""
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        # (batch_size,16,w,h)
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        # (batch_size,24,w,h)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        # (batch_size,24,w,h)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        # (batch_size,24,w,h)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        # (batch_size,88,w,h)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        outputs = torch.cat(outputs, dim=1)

        return outputs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5, stride=1, padding=0)

        self.incep1 = Inception(in_channels=10)
        self.incep2 = Inception(in_channels=20)

        self.fc1 = nn.Linear(5*5*88, 500)  # Linear(输入维度，中间维度)
        self.drop = nn.Dropout2d()
        self.fc2 = nn.Linear(500, 10)  # Linear(中间维度，输出维度)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    '''output_shape = (input_shape - filter_shape + 2*padding) / stride + 1'''

    def forward(self, x):
        # 输入x维度 (N,3,32,32)
        x = self.relu(self.conv1(x))  # 32+1-5 -> (N,10,28,28)
        x = self.maxpool(x)  # 28/2 -> (N,10,14,14)
        x = self.incep1(x)  # (N,10,14,14) -> (N,88,14,14)
        x = self.relu(self.conv2(x))  # 14+1-5 -> (N,20,10,10)
        x = self.maxpool(x)  # 10/2 -> (N,20,5,5)
        x = self.incep2(x)  # (N,20,5,5) -> (N, 88,5,5)
        x = x.view(-1, 5*5*88)  # (N,88,5,5) -> (N,5*5*88)
        x = self.relu(self.fc1(x))  # (N,5*5*88) -> (1,500)
        x = self.drop(x)
        x = self.fc2(x)  # (1,500) -> (1,10)
        return x


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, device, train_dataloader, optimizer, epoch):
    """训练模型"""
    model.train()  # 训练模式 BN有区别
    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        '''forward'''
        y_pred = model(data)
        loss = criterion(y_pred, target)
        '''backward'''
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad.clip_grad_norm_(p, 12)
            # 把梯度下降的速度控制在12以内，防止梯度爆炸
        '''更新参数'''
        optimizer.step()

        if idx % 100 == 0:
            print('Epoch:{}, Iter: {:>d}, Loss: {:>.8f}'.format(
                epoch, idx, loss.item()))


def test(model, device, test_dataloader):
    """测试模型"""
    model.eval()  # 测试模式
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)
            correct += pred.eq(target.view_as(target)).sum().item()

    acc = correct / len(test_dataloader.dataset)
    print('---------------------------')
    print('Accuracy: {:>%}'.format(acc))
    print('---------------------------')


def main():
    for epoch in range(epochs):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)
    torch.save(model.state_dict(), 'mnist_cnn.pt')


if __name__ == "__main__":
    main()
