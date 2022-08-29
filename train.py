# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/29 17:41 
# @Author : wzy 
# @File : train.py
# ---------------------------------------
import math
import torch
from torch import nn
from wzy.Vision_Transformer import data
from arg_parse import parse_args

args = parse_args()


def train(model, device, train_loader, optimizer, loss_func, epoch, epochs):
    model.train()
    trained_samples = 0  # 用于记录已经训练的样本数
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        # 统计已经训练的数据量
        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)  # 返回大于或等于一个给定数字的最小整数

        print('\rTrain epoch: [{}/{}] {}/{} [{}]{}%'.format(epoch, epochs, trained_samples,
                                                            len(train_loader.dataset),
                                                            '-' * progress + '>', progress * 2), end='')


def test(model, device, val_loader, loss_func):
    model.eval()
    test_loss = 0
    num_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            # 输出预测类别
            _, predictions = output.max(1)
            num_correct += (predictions == target).sum()
    test_loss /= len(val_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy:{}/{},({:.4f}%)'.format(
        test_loss.item(), num_correct, len(val_loader.dataset), 100 * num_correct / len(val_loader.dataset)))

    return test_loss, num_correct / len(val_loader.dataset)


def main(model, device, mode):
    # torch.manual_seed(1)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定
    history = []  # 记录loss和acc
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()  # 结合了nn.LogSoftmax()和nn.NLLLoss()两个函数，所以网络最后不需要softmax

    for epoch in range(1, args.epoch + 1):
        train(model, device, data.train_loader, optimizer, loss_func, epoch, args.epoch)
        loss, acc = test(model, device, data.val_loader, loss_func)
        history.append((loss, acc))
    if mode == 1:
        torch.save(model.state_dict(), './pts/vit_model')
    # if mode == 2:
    #     torch.save(model.state_dict(), './pts/quantization_model.pth')
    return model, history