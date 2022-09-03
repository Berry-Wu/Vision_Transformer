# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/3 10:19 
# @Author : wzy 
# @File : MobileViTBlock.py
# ---------------------------------------
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torchsummary import summary


class MLPBlock(nn.Sequential):  # 使用nn.sequential就不用再写forward函数了!!!
    def __init__(self, dim, expansion):
        super().__init__(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(expansion * dim, dim),
            nn.Dropout(0.1)
        )


class ResidualBlock(nn.Sequential):
    def __init__(self, dim, function):
        super(ResidualBlock, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.func = function

    def forward(self, x, **kwargs):
        res = x
        x = self.func(self.ln(x), **kwargs)
        x += res
        return x


class GlobalRepresentation(nn.Module):
    def __init__(self, d_model, L, head):
        super(GlobalRepresentation, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(L):
            self.layers.append(nn.ModuleList([
                ResidualBlock(d_model, Attention(d_model=d_model, head=head)),
                ResidualBlock(d_model, MLPBlock(dim=d_model, expansion=2))
            ]))

    def forward(self, x):
        for att, fnn in self.layers:
            x = att(x)
            x = fnn(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, head=8, dropout=.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // head
        self.d_v = d_model // head
        self.h = head

        # self.attend = nn.Softmax(dim=-1)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.fc_o = nn.Linear(head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        # 分头和qkv的划分(妙啊~)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(q.shape[3])
        attn = torch.softmax(dots, -1)
        out = torch.matmul(attn, v)
        # 多头合并
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        out = self.fc_o(out)
        return out


def feature_and_patch(feature, transformer, p_w, p_h):
    """
    这就和通常的patch划分有区别了，这里得到的patch维度是(Batch Patch Num_patch Channels)
    :param feature: 经过卷积处理的特征图(B,C,H,W)
    :param transformer: transformer块，进行注意力计算
    :param p_w: patch块的w
    :param p_h:patch块的h
    :return:
    """
    _, _, h, w = feature.shape
    patch = rearrange(feature, 'b c (h p1) (w p2) -> b (p1 p2) (h w) c', p1=p_h, p2=p_w)
    patch = transformer(patch)
    feature = rearrange(patch, 'b (p1 p2) (nh nw) c -> b c (nh p1) (nw p2)', nh=h // p_h, nw=w // p_w, p1=p_h, p2=p_w)
    return feature


class MobileViTBlock(nn.Module):
    def __init__(self, in_dim=3, out_dim=192, patch_size=4):
        super(MobileViTBlock, self).__init__()
        self.p_h, self.p_w = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_dim, out_channels=in_dim, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=2 * in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.transformer = GlobalRepresentation(d_model=out_dim, L=2, head=8)

    def forward(self, x):
        residual = x

        # Local Representation
        x = self.conv1(x)
        x = self.conv2(x)

        # Global Representation
        x = feature_and_patch(x, self.transformer, self.p_h, self.p_w)

        # Fusion
        x = self.conv3(x)
        x = torch.cat([x, residual], dim=1)
        x = self.conv4(x)
        return x


if __name__ == '__main__':
    model = MobileViTBlock()
    summary(model, (3, 32, 32), device='cpu')
