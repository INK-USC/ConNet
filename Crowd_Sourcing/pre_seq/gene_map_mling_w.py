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
    parser.add_argument('--output_map', default=None)
    parser.add_argument('--langs', default=None)
    parser.add_argument('--threshold', type=int, default=2)
    parser.add_argument('--unk', default='unk')
    args = parser.parse_args()

    args.langs = args.langs.split()
    print(args.langs)

    gw_map = dict()
    embedding_array = list()
    if args.input_embedding_path != None:
        for lang_id, lang in enumerate(args.langs):
            print(lang_id, lang)
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
            gw_map['<\n>'] = len(gw_map)
            embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])
        if '<\n>' not in gw_map:
            gw_map['<\n>'] = len(gw_map)
            embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])
    else:
        gw_map['<unk>'] = len(gw_map)
        gw_map['<\n>'] = len(gw_map)

    print({'gw_map': len(gw_map), 'emb_array' : len(embedding_array)})
    print(list(gw_map.keys())[:10])

    with open(args.output_map, 'wb') as f:
        pickle.dump({'gw_map': gw_map, 'emb_array': embedding_array}, f)
