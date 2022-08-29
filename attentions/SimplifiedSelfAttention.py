# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/29 18:05 
# @Author : wzy 
# @File : SimplifiedSelfAttention.py
# ---------------------------------------
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import init


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        # wzy改进
        self.qkv = nn.Linear(d_model, d_model * 3)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        # wzy改进，仅输入一次x即可，便于在sequential中构造注意力块
        qkv = rearrange(self.qkv(x), "b n (d qkv) -> (qkv) b n d", qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            # masked_fill(mask, value)
            # 用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        # 这里同时将多个头进行了合并
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


if __name__ == '__main__':
    input = torch.randn(3, 197, 768)
    ssa = SimplifiedScaledDotProductAttention(d_model=768, h=8)
    output = ssa(input)
    print(output.shape)
