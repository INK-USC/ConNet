# For more info about the data go to:
# http://conll.cemantix.org/2012/data.html
# https://natural-language-understanding.fandom.com/wiki/OntoNotes

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


def list_files(data_dir):
    genres = os.listdir(data_dir)
    genre_data = []
    for g in genres:
        if g == 'pt': # pt portion doesn't contain any named entities
            continue
        data_dir_t = data_dir+'/'+g
        data_folders1 = [data_dir_t+'/'+f for f in os.listdir(data_dir_t)]
        data_folders2 = [f for ff in [[f1+'/'+r for r in os.listdir(f1)] for f1 in data_folders1] for f in ff]
        data_files = [f for ff in [[f2+'/'+r for r in os.listdir(f2)] for f2 in data_folders2] for f in ff]
        data_files_gold = [r for r in data_files if r.endswith('.gold_conll')]
        assert len(data_files_gold) == len(data_files) / 2
        genre_data.append([g, data_files_gold])
    
    return genre_data


def read_on(genre_data):
    data = []
    curr_words, curr_pos, curr_net, curr_doc_part, curr_ne = [], [], [], None, None
    data_id = 0
    for genre, data_files in genre_data:
        for file in data_files:
            with open(file, 'r') as f:
                line = f.readline()
                assert line.split(' ')[2].split('/')[0][1:] == genre # make sure genre is correct
                for line in f:
                    if line.startswith('#end document'):
                        assert curr_ne == None
                        break
                    if line == '\n':
                        data.append({'data_id': data_id, 'doc_part': curr_doc_part, 'genre': genre, \
                                     'words': curr_words, 'pos_tags': curr_pos, 'ne': curr_net})
                        curr_words, curr_pos, curr_net, curr_doc_part, curr_ne = [], [], [], None, None
                        data_id += 1
                        continue
                    split = [r for r in line.strip().split(' ') if not r in ['', None]]
                    if curr_doc_part == None:
                        curr_doc_part = split[0] + ' - ' + split[1]
                    curr_words.append(split[3])
                    curr_pos.append(split[4])
                    ne = split[10].replace('*', '')
                    if ne == '':
                        if curr_ne:
                            curr_net.append('I-'+curr_ne)
                        else:
                            curr_net.append('O')
                    elif '(' in ne and ')' in ne:
                        assert curr_ne == None
                        curr_net.append('B-' + ne[1:-1])
                    elif '(' in ne:
                        assert curr_ne == None
                        curr_net.append('B-' + ne[1:])
                        curr_ne = ne[1:]
                    elif ')' in ne:
                        assert curr_ne
                        curr_net.append('I-' + curr_ne)
                        curr_ne = None
    
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

def create_feats(data, char2idx, pos2idx, task2idx, label2idx):
    for i in range(len(data)):
        d = data[i]
        data_id = d['data_id']
        doc_part = d['doc_part']
        task = task2idx[d['genre']]
        words = d['words']
        char_idx, char_boundaries = char_feats(words, char2idx)
        pos_tags = [pos2idx[r] for r in d['pos_tags']]
        label = [label2idx[r] for r in d['ne']]
        feats = {'data_id': data_id, 'doc_part': doc_part, 'task': task, 'chars': char_idx, \
                 'char_boundaries': char_boundaries, 'words': words, 'pos_tags': pos_tags, 'label': label}
        data[i] = {'ref': d, 'feats': feats}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', help='path to data directory')
    p.add_argument('-word_emb_dir', help='path to word embedding files')
    p.add_argument('-save_data', help='path to save processed data')
    args = p.parse_args()
    
    print(args)
    
    # For debugging
    # pickle.dump(args, open('pickle_breakpoints/read_onto_notes.p', 'wb'))
    # assert False
    # args = pickle.load(open('pickle_breakpoints/read_onto_notes.p', 'rb'))
    
    train_genre_data = list_files(args.data_dir + '/train/data/english/annotations')
    dev_genre_data = list_files(args.data_dir + '/development/data/english/annotations')
    test_genre_data = list_files(args.data_dir + '/test/data/english/annotations')
    
    train_data = read_on(train_genre_data)
    dev_data = read_on(dev_genre_data)
    test_data = read_on(test_genre_data)
    
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
    
    all_pos = ['<pad>'] + list(set([r for rr in train_data for r in rr['pos_tags']]))
    pos2idx = {k:v for v,k in enumerate(all_pos)}
    all_labels = ['<start>', '<pad>'] + list(set([r for rr in train_data for r in rr['ne']]))
    label2idx = {k:v for v,k in enumerate(all_labels)}
    all_tasks = list(set([r['genre'] for r in test_data])) # here "task" means genre
    task2idx = {k:v for v,k in enumerate(all_tasks)}
    pickle.dump({'pos2idx': pos2idx, 'label2idx': label2idx, 'task2idx': task2idx}, open(args.save_data+'/common.p', 'wb'))
    
    create_feats(train_data, char2idx, pos2idx, task2idx, label2idx)
    create_feats(dev_data, char2idx, pos2idx, task2idx, label2idx)
    create_feats(test_data, char2idx, pos2idx, task2idx, label2idx)
    
    pickle.dump({'train_data': train_data, 'dev_data': dev_data, 'test_data': test_data}, open(args.save_data+'/data.p', 'wb'))
    