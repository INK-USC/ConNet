"""
.. module:: utils
    :synopsis: utility tools
 
.. moduleauthor:: Liyuan Liu, Frank Xu
"""

import codecs
import csv
import itertools
from functools import reduce

import numpy as np
import shutil
import torch
import json

import torch.nn as nn
import torch.nn.init

zip = getattr(itertools, 'izip', zip)


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum

    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def switch(vec1, vec2, mask):
    """
    switch function for pytorch

    args:
        vec1 (any size) : input tensor corresponding to 0
        vec2 (same to vec1) : input tensor corresponding to 1
        mask (same to vec1) : input tensor, each element equals to 0/1
    return:
        vec (*)
    """
    catvec = torch.cat([vec1.view(-1, 1), vec2.view(-1, 1)], dim=1)
    switched_vec = torch.gather(catvec, 1, mask.long().view(-1, 1))
    return switched_vec.view(-1)


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

