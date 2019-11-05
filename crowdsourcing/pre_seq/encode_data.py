"""
.. module:: encode_data
    :synopsis: encode data for sequence labeling
 
.. moduleauthor:: Liyuan Liu
"""
import pickle
import argparse
import os, sys
import random
import numpy as np

from tqdm import tqdm

import itertools
import functools
from collections import Counter

def encode_dataset(input_file, gw_map, c_map, y_map):

    gw_unk = gw_map['<unk>']
    c_con = c_map[' ']
    c_unk = c_map['<unk>']
    if '<unk>' not in y_map:
        y_map['<unk>'] = len(y_map)

    dataset = list()

    tmpw_gw, tmpc, tmpy = list(), list(), list()

    with open(input_file, 'r') as fin:
        for line in fin:
            if line.isspace() or line.startswith('-DOCSTART-'):
                if len(tmpw_gw) > 0:
                    dataset.append([tmpw_gw, tmpc, tmpy])
                tmpw_gw, tmpc, tmpy = list(), list(), list()
            else:
                line = line.split()
                tmpw_gw.append(gw_map.get(line[0].lower(), gw_unk))
                tmpy.append(y_map.get(line[-1], c_unk))
                tmpc.append([c_map.get(tup, c_unk) for tup in line[0]])

    if len(tmpw_gw) > 0:
        dataset.append([tmpw_gw, tmpc, tmpy])

    return dataset

def encode_dataset_ma(input_file, gw_map, c_map, y_map, task = 'ma'):

    gw_unk = gw_map['<unk>']
    c_con = c_map[' ']
    c_unk = c_map['<unk>']
    #y_map['<unk>'] = len(y_map)
    if '<unk>' not in y_map:
        y_map['<unk>'] = len(y_map)
    y_unk = y_map['<unk>']

    dataset = list()

    tmpw_gw, tmpc, tmpy = list(), list(), list()

    with open(input_file, 'r') as fin:
        for line in fin:
            if line.isspace() or line.startswith('-DOCSTART-'):
                if len(tmpw_gw) > 0:
                    if task == 'vote_tok':
                        #tmpy = [max(set(y), key=y.count) for y in tmpy]    
                        tmp = []
                        for y in tmpy:
                            counts = Counter(y)
                            l = counts.most_common(2)
                            if l[0][0] != y_unk:
                                tmp += [l[0][0]]
                            elif len(l) == 1:
                                tmp += [y_map['O']]
                            else:
                                tmp += [l[1][0]]
                        tmpy = tmp
                    elif task == 'vote_seq':
                        #tmpy = [max(set(y), key=y.count) for y in tmpy]
                        toky = []
                        for y in tmpy:
                            counts = Counter(y)
                            l = counts.most_common(2)
                            if l[0][0] != y_unk:
                                toky += [l[0][0]]
                            elif len(l) == 1:
                                toky += [y_map['O']]
                            else:
                                toky += [l[1][0]]
                        scorey = [0]*len(tmpy[0])
                        for i, l in enumerate(tmpy):
                            for j in range(len(l)):
                                if toky[i] == l[j]:
                                    scorey[j] += 1
                        maj_idx = scorey.index(max(scorey))
                        tmpy = [y[maj_idx] for y in tmpy]
                        tmpy = [y_map['O'] if y == y_unk else y for y in tmpy]
                    dataset.append([tmpw_gw, tmpc, tmpy])
                tmpw_gw, tmpc, tmpy = list(), list(), list()
            else:
                line = line.split()
                tmpw_gw.append(gw_map.get(line[0].lower(), gw_unk))
                tmpc.append([c_map.get(tup, c_unk) for tup in line[0]])
                tmpy.append([y_map[y] for y in line[1:]])

    if len(tmpw_gw) > 0:
        """
        if task == 'vote_tok':
            tmpy = [max(set(y), key=y.count) for y in tmpy]    
        elif task == 'vote_seq':
            toky = [max(set(y), key=y.count) for y in tmpy]
            scorey = [0]*len(tmpy[0])
            for i, l in enumerate(tmpy):
                for j in range(len(l)):
                    if toky[i] == l[j]:
                        scorey[j] += 1
            maj_idx = scorey.index(max(scorey))
            tmpy = [y[maj_idx] for y in tmpy]
        """
        if task == 'vote_tok':
            #tmpy = [max(set(y), key=y.count) for y in tmpy]    
            tmp = []
            for y in tmpy:
                counts = Counter(y)
                l = counts.most_common(2)
                if l[0][0] != y_unk:
                    tmp += [l[0][0]]
                elif len(l) == 1:
                    tmp += [y_map['O']]
                else:
                    tmp += [l[1][0]]
            tmpy = tmp
        elif task == 'vote_seq':
            #tmpy = [max(set(y), key=y.count) for y in tmpy]
            toky = []
            for y in tmpy:
                counts = Counter(y)
                l = counts.most_common(2)
                if l[0][0] != y_unk:
                    toky += [l[0][0]]
                elif len(l) == 1:
                    toky += [y_map['O']]
                else:
                    toky += [l[1][0]]
            scorey = [0]*len(tmpy[0])
            for i, l in enumerate(tmpy):
                for j in range(len(l)):
                    if toky[i] == l[j]:
                        scorey[j] += 1
            maj_idx = scorey.index(max(scorey))
            tmpy = [y[maj_idx] for y in tmpy]
        dataset.append([tmpw_gw, tmpc, tmpy])

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="../DDCLM/data/ner/eng.train.iobes")
    parser.add_argument('--test_file', default="../DDCLM/data/ner/eng.testb.iobes")
    parser.add_argument('--dev_file', default="../DDCLM/data/ner/eng.testa.iobes")
    parser.add_argument('--input_map', default="./data/conll_map.pk")
    parser.add_argument('--output_file', default="./data/ner_dataset.pk")
    parser.add_argument('--unk', default='<unk>')
    parser.add_argument('--task', default='true', choices=['true', 'vote_seq', 'vote_tok', 'ma'])
    args = parser.parse_args()

    with open(args.input_map, 'rb') as f:
        p_data = pickle.load(f)
        name_list = ['gw_map', 'c_map', 'y_map', 'emb_array']
        gw_map, c_map, y_map, emb_array = [p_data[tup] for tup in name_list]

    bias = 2 * np.sqrt(3.0 / len(emb_array[0]))
    if '<unk>' not in gw_map:
        gw_map['<unk>'] = len(gw_map)
        emb_array.append([random.random() * bias - bias for tup in emb_array[0]])

    if args.task == 'true':
        train_dataset = encode_dataset(args.train_file, gw_map, c_map, y_map)
        test_dataset = encode_dataset(args.test_file, gw_map, c_map, y_map)
        dev_dataset = encode_dataset(args.dev_file, gw_map, c_map, y_map)
    else:
        train_dataset = encode_dataset_ma(args.train_file, gw_map, c_map, y_map, task = args.task)
        test_dataset = encode_dataset(args.test_file, gw_map, c_map, y_map)
        dev_dataset = encode_dataset(args.dev_file, gw_map, c_map, y_map)


    print({'train_data': len(train_dataset), 'test_data': len(test_dataset), 'dev_data': len(dev_dataset)})
    with open(args.output_file, 'wb') as f:
        pickle.dump({'gw_map': gw_map, 'c_map': c_map, 'y_map': y_map, 'emb_array': emb_array, 'train_data': train_dataset, 'test_data': test_dataset, 'dev_data': dev_dataset}, f)
        
