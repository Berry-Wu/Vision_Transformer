# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/29 17:40 
# @Author : wzy 
# @File : data.py
# ---------------------------------------
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from arg_parse import parse_args

args = parse_args()

train_data = torchvision.datasets.CIFAR10(
    root='../data/',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True
)

val_data = torchvision.datasets.CIFAR10(
    root='../data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True
)

train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=True)

if __name__ == '__main__':
    print(train_data.data[0].shape)