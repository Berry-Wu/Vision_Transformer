# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/29 17:49 
# @Author : wzy 
# @File : visual.py
# ---------------------------------------
from matplotlib import pyplot as plt


def draw(history, epochs):
    # 模型的loss和acc分析
    x = list(range(1, epochs + 1))

    plt.subplot(1, 1, 1)
    plt.plot(x, [history[i][1] for i in range(epochs)], label='vit')

    plt.title('Test accuracy')
    plt.legend()

    plt.savefig("visual.png")