# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/25 15:11 
# @Author : wzy 
# @File : arg_parse.py
# ---------------------------------------
import argparse


def parse_args():
    parse = argparse.ArgumentParser(description="The hyper-parameter of ViT")
    parse.add_argument('-b', '--bs', default=128)
    parse.add_argument('-l', '--lr', default=1e-3)
    parse.add_argument('-e', '--epoch', default=20)

    args = parse.parse_args()
    return args
