from pathlib import Path
import pickle
import sys, os
import argparse
from collections import defaultdict, Counter, OrderedDict
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


# read glove embeddings
def read_emb(emb_file, word_vocab):
    with open(emb_file, 'r', encoding="utf-8") as f:
        emb_list = []
        for x in f:
            xx = x.strip().split(' ')
            if xx[0] in word_vocab:
                emb_list.append((xx[0], [float(r) for r in xx[1:]]))
    
    word_emb = np.array([r[1] for r in emb_list])
    # add special tokens
    word_emb = np.vstack((np.zeros((1, word_emb.shape[1])), word_emb))
    word_emb = np.vstack((np.mean(word_emb, 0), word_emb))
    curr_word_vocab = ['<unk>', '<pad>'] + [r[0] for r in emb_list]
    
    word2idx = OrderedDict((curr_word_vocab[i], i) for i in range(len(curr_word_vocab)))
    
    return (word_emb, word2idx)


def read_gum(data_file):
    data, curr_data = [], []
    curr_doc_id, curr_genre, curr_sent_id = None, None, None
    with open(data_file, 'r') as f:
        i = 0
        for line in f:
            if line.startswith('# newdoc id = '):
                curr_doc_id = line.strip().split('# newdoc id = ')[1]
                _, curr_genre, detail = curr_doc_id.split('_')
            elif line.startswith('# sent_id = '):
                curr_sent_id = line.strip().split('# sent_id = ')[1]
            elif line == '\n':
                data.append({'data_id': i, 'doc_id': curr_doc_id, 'sent_id': curr_sent_id, 'genre': curr_genre, \
                             'words': [r[0] for r in curr_data], 'pos_tags': [r[1] for r in curr_data]})
                curr_data = []
                i += 1
            elif line[0] != '#':
                id, form, lemma, upostag, xpostag, feats, head, deprel, deps, misc = line.strip().split('\t')
                curr_data.append([form, upostag])
    
    return data


def char_feats(words, char2idx):
    char_idx, char_boundaries, curr_char_boundary = [], [], []
    posit = 0
    for i in range(len(words)):
        if i != 0:
            char_idx.append(char2idx['<dlm>'])
            posit += 1
        curr_char_boundary.append(posit)
        for c in words[i]:
            char_idx.append(char2idx[c.lower()] if c.lower() in char2idx else char2idx['<unk>'])
            posit += 1
        curr_char_boundary.append(posit-1)
        char_boundaries.append(curr_char_boundary)
        curr_char_boundary = []
    
    return char_idx, char_boundaries

def create_feats(data, char2idx, word2idx, task2idx, label2idx):
    for i in range(len(data)):
        d = data[i]
        data_id = d['data_id']
        task = task2idx[d['genre']]
        words = d['words']
        char_idx, char_boundaries = char_feats(words, char2idx)
        label = [label2idx[r] for r in d['pos_tags']]
        feats = {'data_id': data_id, 'task': task, 'chars': char_idx, \
                 'char_boundaries': char_boundaries, 'words': words, 'label': label}
        data[i] = {'ref': d, 'feats': feats}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', help='path to data directory')
    p.add_argument('-word_emb_dir', help='path to word embedding files')
    p.add_argument('-save_data', help='path to save processed data')
    args = p.parse_args()
    
    print(args)
    
    # For debugging
    # pickle.dump(args, open('pickle_breakpoints/read_ud.p', 'wb'))
    # assert False
    # args = pickle.load(open('pickle_breakpoints/read_ud.p', 'rb'))
    
    train_data = read_gum(args.data_dir + '/en_gum-ud-train.conllu')
    dev_data = read_gum(args.data_dir + '/en_gum-ud-dev.conllu')
    test_data = read_gum(args.data_dir + '/en_gum-ud-test.conllu')
    
    all_words = [r.lower() for rr in train_data+dev_data+test_data for r in rr['words']]
    word_vocab = list(set(all_words))
    for word_emb_file in ['glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt']:
        word_emb, word2idx = read_emb(args.word_emb_dir+'/' + word_emb_file, word_vocab)
        pickle.dump({'word_emb': word_emb, 'word2idx': word2idx}, open(args.save_data + '/' + word_emb_file + '.p', 'wb'))
    
    # leave 10% chars as OOV
    train_chars = [r for r in ''.join([r.lower() for rr in train_data for r in rr['words']])]
    train_chars = [r[0] for r in sorted(Counter(train_chars).items(), key=lambda x:x[1], reverse=True)]
    train_chars = ['<unk>', '<pad>', '<dlm>'] + train_chars[:int(len(train_chars)*0.9)] # dlm is word delimiter
    char2idx = {y:x for x,y in enumerate(train_chars)}
    for char_emb_dim in [30,50,100,200]:
        char_emb = nn.Embedding(len(char2idx), char_emb_dim, padding_idx=char2idx['<pad>'])
        char_emb = char_emb.weight.detach().numpy()
        pickle.dump({'char_emb': char_emb, 'char2idx': char2idx}, open(args.save_data + '/random_char_emb_' + str(char_emb_dim) + '.p', 'wb'))
    
    all_labels = ['<start>', '<pad>'] + list(set([r for rr in test_data for r in rr['pos_tags']]))
    label2idx = {k:v for v,k in enumerate(all_labels)}
    all_tasks = list(set([r['genre'] for r in test_data])) # here "task" means genre
    task2idx = {k:v for v,k in enumerate(all_tasks)}
    pickle.dump({'label2idx': label2idx, 'task2idx': task2idx}, open(args.save_data+'/common.p', 'wb'))
    
    create_feats(train_data, char2idx, word2idx, task2idx, label2idx)
    create_feats(dev_data, char2idx, word2idx, task2idx, label2idx)
    create_feats(test_data, char2idx, word2idx, task2idx, label2idx)
    
    pickle.dump({'train_data': train_data, 'dev_data': dev_data, 'test_data': test_data}, open(args.save_data+'/data.p', 'wb'))
    