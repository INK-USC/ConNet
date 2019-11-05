"""
.. module:: gene_map
    :synopsis: generate map for sequence labeling
 
.. moduleauthor:: Liyuan Liu
"""
import pickle
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import sys

import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_corpus_path', default=None)
    parser.add_argument('--input_embedding_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--lang', default=None)
    parser.add_argument('--threshold', type=int, default=2)
    parser.add_argument('--unk', default='unk')
    args = parser.parse_args()

    #args.langs = args.langs.split()
    #print(args.langs)
    lang = args.lang

    gw_map = dict()
    embedding_array = list()
    if args.input_embedding_path != None:
        #for lang_id, lang in enumerate(args.langs):
        #    print(lang_id, lang)
        #lang='af'
        input_embedding_file = os.path.join(args.input_embedding_path, lang+'2en/vectors-'+lang+'.txt')
        headline = True
        for line in open(input_embedding_file, 'r', encoding="utf-8"):
            if headline:
                headline = False
                continue
            line = line.split()
            word = lang + ':' + line[0]
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            if line[0] == args.unk:
                gw_map['<unk>'] = len(gw_map)
            else:
                gw_map[word] = len(gw_map)
            embedding_array.append(vector)

        bias = 2 * np.sqrt(3.0 / len(embedding_array[0]))

        if '<unk>' not in gw_map:
            gw_map['<unk>'] = len(gw_map)
            embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])
        if '<\n>' not in gw_map:
            gw_map['<\n>'] = len(gw_map)
            embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])
    else:
        gw_map['<unk>'] = len(gw_map)
        gw_map['<\n>'] = len(gw_map)

    w_count = dict()
    c_count = dict()
    y_map = dict()

    #for lang in args.langs:
    train_corpus = os.path.join(args.train_corpus_path, lang+'/train')
    with open(train_corpus, 'r') as fin:
        for line in fin:
            if line.isspace() or line.startswith('-DOCSTART-'):
                c_count['\n'] = c_count.get('\n', 0) + 1
            else:
                line = line.split()
                for tup in line[0]:
                    tup = lang + ':' + tup
                    c_count[tup] = c_count.get(tup, 0) + 1
                c_count[' '] = c_count.get(' ', 0) + 1
                if line[-1] not in y_map:
                    y_map[line[-1]] = len(y_map)


    c_map = {}
    for k, v in enumerate(c_count.items()):
        if v[1] > args.threshold:
            c_map[v[0]] = len(c_map)
    c_map['<unk>'] = len(c_map)

    y_map['<s>'] = len(y_map)
    y_map['<eof>'] = len(y_map)

    print({'gw_map': len(gw_map), 'c_map': len(c_map), 'y_map': len(y_map), 'emb_array' : len(embedding_array)})
    print(list(gw_map.keys())[:10])
    print(c_map)
    print(y_map)

    output_map = os.path.join(args.output_path, lang+'/map.pk')
    with open(output_map, 'wb') as f:
        pickle.dump({'gw_map': gw_map, 'c_map': c_map, 'y_map': y_map, 'emb_array': embedding_array}, f)
