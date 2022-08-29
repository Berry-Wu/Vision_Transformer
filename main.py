# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/29 17:42 
# @Author : wzy 
# @File : main.py
# ---------------------------------------
import numpy as np
import torch
from arg_parse import parse_args
from wzy.Vision_Transformer.models.ViT import ViT
import train
from visual import draw

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = ViT().to(device)
    _, history = train.main(model, device, 1)

    history = np.array(torch.tensor(history, device='cpu'))
    draw(history, args.epoch)
