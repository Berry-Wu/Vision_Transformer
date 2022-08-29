# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/29 14:42
# @Author : wzy 
# @File : ViT.py
# @reference:https://github.com/FrancescoSaverioZuppichini/ViT
# ---------------------------------------
import torch.nn as nn
import torch
from wzy.Vision_Transformer.attentions.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from einops.layers.torch import Rearrange, Reduce
from einops import repeat
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, dim=192, img_size=32):
        super(PatchEmbedding, self).__init__()
        self.patch_division = nn.Sequential(
            # 1.divide the img to patchs
            Rearrange('b c (w s1) (h s2) -> b (w h) (c s1 s2)', s1=patch_size, s2=patch_size),
            # 2.linear projection for learnable parameter
            nn.Linear(in_channels * patch_size * patch_size, dim)
        )
        # 3.cls_token: one-dimensional and length equals to dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 4.position:the position of patch in the img
        self.position = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_division(x)
        # 一个batch中的每个图片都需要生成一个cls_token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        # 将cls_token拼接到tokens最前面
        x = torch.cat([cls_tokens, x], dim=1)
        # 添加位置信息
        x += self.position
        return x


class MLPBlock(nn.Sequential):  # 使用nn.sequential就不用再写forward函数了
    def __init__(self, dim, expansion):
        super().__init__(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(expansion * dim, dim),
        )


class ResidualBlock(nn.Sequential):
    def __init__(self, function):
        super(ResidualBlock, self).__init__()
        self.func = function

    def forward(self, x, **kwargs):
        res = x
        x = self.func(x, **kwargs)
        x += res
        return x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, dim=192, expansion=4):
        super().__init__(
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(dim),
                SimplifiedScaledDotProductAttention(dim, h=3),
                nn.Dropout(0.1),
            )),
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(dim),
                MLPBlock(dim, expansion),
                nn.Dropout(0.1),
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, num_block=4, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(num_block)])


class MLPHead(nn.Sequential):
    def __init__(self, dim, num_cls):
        super().__init__(
            Reduce('b n d -> b d', reduction='mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_cls)
        )


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels=3,
                 patch_size=4,
                 dim=192,
                 img_size=32,
                 num_block=4,
                 num_cls=10,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, dim, img_size),
            TransformerEncoder(num_block, **kwargs),
            MLPHead(dim, num_cls)
        )


if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    patch_embedded = PatchEmbedding()(input)
    encoder = TransformerEncoderBlock()(patch_embedded)
    print(encoder.shape)
    summary(ViT(), (3, 32, 32), device='cpu')
