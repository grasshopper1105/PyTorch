import torch
import visdom
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.ResNet.ResNet import ResNet


def main():

    batchsz = 128  # 一次读入128张照片

    # 数据导入和增强
    cifar_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.Resize([32, 32]),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ]), download=True)  # 每次从训练集读入一张照片并进行数据增强
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True, num_workers=2)  # 按批次读入

    cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ]), download=True)  # 每次从测试集集读入一张照片
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True, num_workers=2)  # 按批次读入

    x_train, y_train = iter(cifar_train).next()  # 得到训练集的一个batch
    x_test, y_test = iter(cifar_test).next()  # 得到测试集的一个batch
    print('x_train.shape is :', x_train.shape, '\ty_train.shape is :', y_train.shape)  # 打印训练集形状
    print('x_test.shape is :', x_test.shape, '\ty_test.shape is :', y_test.shape)

    viz = visdom.Visdom()
    device = torch.device('cuda')
    model = ResNet().to(device)  # 导入模型并放到GPU上计算
    criteon = nn.CrossEntropyLoss().to(device)  # 交叉熵计算LOSS
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam优化器
    print(model)

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))  # 绘制LOSS曲线
    viz.line([0], [-1], win='acc', opts=dict(title='acc'))  # 绘制ACC曲线

    # train
    for epoch in range(10):

        model.train()  # train模式
        for batchidx, (x_train, y_train) in enumerate(cifar_train):
            x_train, y_train = x_train.to(device), y_train.to(device)  # 将数据集转移到GPU
            logits = model(x_train)
            loss = criteon(logits, y_train)

            # backprop
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度并累加
            for p in model.parameters():
                torch.nn.utils.clip_grad.clip_grad_norm_(p, 12)  # 把梯度下降的速度控制在12以内，防止梯度爆炸
            optimizer.step()  # 每个step走一遍过程

            viz.line([loss.item()], [global_step], win='loss', update='append')  # 绘制loss
            global_step += 1

        print('-------------------------------------------------------')
        print('now the epoch is:', epoch, '\nthe loss is:', loss.item())

        model.eval()  # test模式
        with torch.no_grad():  # 不需要backprop
            # test
            total_correct = 0
            total_num = 0
            for x_test, y_test in cifar_test:
                x_test, y_test = x_test.to(device), y_test.to(device)  # 将数据转移到GPU
                logits = model(x_test)
                # [b,10] -> [b]
                pred = logits.argmax(dim=1)  # 选出第一维上最大数据的索引
                total_correct += torch.eq(pred, y_test).float().sum().item()  # 统计预测正确的数目
                total_num += x_test.size(0)  # 当前数据总数

            acc = total_correct / total_num  # 准确率
            print('the accurancy is:%.2f%%' % (100 * acc))

            if acc > best_acc:
                best_epoch = epoch
                best_acc = acc

                torch.save(model.state_dict(), 'best.mdl')  # 保存最优模型
                viz.line([acc], [global_step], win='acc', update='append')  # 绘制acc

    print('the best accurancy is:%.2f%%' % (100 * best_acc))

    model.state_dict(torch.load('best.mdl'))
    print('----------loader from ckpt----------')


if __name__ == '__main__':
    main()
