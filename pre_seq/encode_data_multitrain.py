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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="../DDCLM/data/ner/eng.train.iobes")
    parser.add_argument('--a_num', default=47, type=int, help='the number of annotators (train files)')
    parser.add_argument('--test_file', default="../DDCLM/data/ner/eng.testb.iobes")
    parser.add_argument('--dev_file', default="../DDCLM/data/ner/eng.testa.iobes")
    parser.add_argument('--input_map', default="./data/conll_map.pk")
    parser.add_argument('--output_file', default="./data/ner_dataset.pk")
    parser.add_argument('--unk', default='<unk>')
    args = parser.parse_args()

    with open(args.input_map, 'rb') as f:
        p_data = pickle.load(f)
        name_list = ['gw_map', 'c_map', 'y_map', 'emb_array']
        gw_map, c_map, y_map, emb_array = [p_data[tup] for tup in name_list]

    train_datasets = []
    for i in range(args.a_num):
        train_file = os.path.join(args.train_path, str(i)) 
        train_dataset = encode_dataset(train_file, gw_map, c_map, y_map)
        train_datasets.append(train_dataset)
    test_dataset = encode_dataset(args.test_file, gw_map, c_map, y_map)
    dev_dataset = encode_dataset(args.dev_file, gw_map, c_map, y_map)


    print({'train_datas': len(train_datasets), 'test_data': len(test_dataset), 'dev_data': len(dev_dataset)})
    with open(args.output_file, 'wb') as f:
        pickle.dump({'gw_map': gw_map, 'c_map': c_map, 'y_map': y_map, 'emb_array': emb_array, 'train_datas': train_datasets, 'test_data': test_dataset, 'dev_data': dev_dataset}, f)
        
