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

    dataset = list()

    tmpw_gw, tmpc, tmpy = list(), list(), list()

    oov, vocab = 0,0

    with open(input_file, 'r') as fin:
        for line in fin:
            if line.isspace() or line.startswith('-DOCSTART-'):
                if len(tmpw_gw) > 0:
                    dataset.append([tmpw_gw, tmpc, tmpy])
                tmpw_gw, tmpc, tmpy = list(), list(), list()
            else:
                line = line.split()
                tmpw_gw.append(gw_map.get(line[0].lower(), gw_unk))
                tmpy.append(y_map.get(line[-1]))
                tmpc.append([c_map.get(tup, c_unk) for tup in line[0]])

                if tmpw_gw[-1] == gw_unk:
                    oov += 1
                vocab += 1

    if len(tmpw_gw) > 0:
        dataset.append([tmpw_gw, tmpc, tmpy])

    return dataset, float(oov) / vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', default="../DDCLM/data/ner/eng.train.iobes")
    parser.add_argument('--input_map', default="./data/conll_map.pk")
    parser.add_argument('--output_file', default="./data/ner_dataset.pk")
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--unk', default='<unk>')
    parser.add_argument('--langs', default=None)
    args = parser.parse_args()

    with open(args.input_map, 'rb') as f:
        p_data = pickle.load(f)
        name_list = ['gw_map', 'c_map', 'y_map', 'emb_array']
        gw_map, c_map, y_map, emb_array = [p_data[tup] for tup in name_list]

    if args.log_file != None:
        log_file = open(args.log_file, 'w')

    log_file.write(args.langs+'\n')
    args.langs = args.langs.split()
    
    train_datasets, test_datasets, dev_datasets = {}, {}, {}
    for lang in args.langs:
        train_file = os.path.join(args.train_file_path, lang+'/train')
        dev_file = os.path.join(args.train_file_path, lang+'/dev')
        test_file = os.path.join(args.train_file_path, lang+'/test')

        train_datasets[lang], train_oov = encode_dataset(train_file, gw_map, c_map, y_map)
        test_datasets[lang], test_oov = encode_dataset(test_file, gw_map, c_map, y_map)
        dev_datasets[lang], dev_oov = encode_dataset(dev_file, gw_map, c_map, y_map)

        log_file.write(lang + ': {train_data: ' + str(len(train_datasets[lang])) + '; test_data: ' + str(len(test_datasets[lang])) + '; dev_data: ' + str(len(dev_datasets[lang])) + '}\n')
        log_file.write('OOV: '+ str(train_oov) + '; ' + str(test_oov) + '; ' + str(dev_oov) + '\n')

    with open(args.output_file, 'wb') as f:
        pickle.dump({'train_data': train_datasets, 'test_data': test_datasets, 'dev_data': dev_datasets}, f)
        
