from pathlib import Path
import pickle
import sys, os
import argparse
from collections import defaultdict, Counter, OrderedDict
from functools import reduce
import logging as log
import numpy as np
import random
import math, re
import time
import copy
import torch
import torch.nn as nn

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)


def read_data(path):
    data = []
    i = 0
    for data_file in ['positive.review', 'negative.review']:
        label = 1 if data_file == 'positive.review' else 0
        with open(path+'/'+data_file, 'r') as f:
            for line in f:
                d = [r.split(':') for r in line.strip().split(' ')]
                assert d[-1][0] == '#label#'
                d = d[:-1]
                d = dict([[r[0], int(r[1])] for r in d])
                data.append({'data_id': i, 'word_freq': d, 'label': label})
    
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', help='path to data directory')
    p.add_argument('-word_emb_dir', help='path to word embedding files')
    p.add_argument('-save_data', help='path to save processed data')
    args = p.parse_args()
    
    print(args)
    
    # For debugging
    # pickle.dump(args, open('pickle_breakpoints/read_senti_data.p', 'wb'))
    # assert False
    # args = pickle.load(open('pickle_breakpoints/read_senti_data.p', 'rb'))
    
    data = []
    i = 0
    for data_dir in ['books', 'dvd', 'electronics', 'kitchen']:
        for data_file in ['positive.review', 'negative.review']:
            label = 1 if data_file == 'positive.review' else 0
            with open(args.data_dir+'/'+data_dir+'/'+data_file, 'r') as f:
                for line in f:
                    d = [r.split(':') for r in line.strip().split(' ')]
                    assert d[-1][0] == '#label#'
                    d = d[:-1]
                    d = dict([[r[0], int(r[1])] for r in d])
                    data.append({'ref': {'data_id': i, 'word_freq': d, 'label': label, 'task': data_dir}})
                    i += 1
    
    task2idx = dict([[r[1], r[0]] for r in enumerate(set([r['ref']['task'] for r in data]))])
    word_freq_tot = reduce(lambda x,y:x+y, [Counter(r['ref']['word_freq']) for r in data])
    selected_words = [r[0] for r in word_freq_tot.most_common(5000)]
    word2idx = dict([[r[1], r[0]] for r in enumerate(selected_words)])
    pickle.dump({'word2idx': word2idx, 'task2idx': task2idx}, open(args.save_data+'/common.p', 'wb'))
    
    for i in range(len(data)):
        ref = data[i]['ref']
        word_feat = np.zeros(5000)
        for w, count in ref['word_freq'].items():
            if w in word2idx:
                word_feat[word2idx[w]] = count
        
        feats = {'data_id': ref['data_id'], 'word_feat': word_feat, 'label': ref['label'], 'task': task2idx[ref['task']]}
        data[i]['feats'] = feats
    
    pickle.dump({'data': data}, open(args.save_data+'/data.p', 'wb'))
    