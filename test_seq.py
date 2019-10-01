from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_seq.crf import CRFLoss, CRFDecode
from model_seq.dataset import SeqDataset
from model_seq.evaluator_with_saveres import eval_wc
from model_seq.seqlabel import Vanilla_SeqLabel
from model_seq.seqlabel import SeqLabel_CRF
import model_seq.utils as utils

from torch_scope import wrapper

import argparse
import json
import os
import sys
import itertools
import functools
import random
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='/home/ron_data/ouyu/exp/CN_NER/checkpoint')
    parser.add_argument('--checkpoint_name', default='ner')
    parser.add_argument('--git_tracking', action='store_true')
    parser.add_argument('--restore_model_path', default=None)

    parser.add_argument('--corpus', default='./data/ner_dataset.pk')
    parser.add_argument('--corpus_test', default='./data/ner_dataset.pk')

    parser.add_argument('--seq_c_dim', type=int, default=30)
    parser.add_argument('--seq_c_hid', type=int, default=150)
    parser.add_argument('--seq_c_layer', type=int, default=1)
    parser.add_argument('--seq_w_dim', type=int, default=100)
    parser.add_argument('--seq_w_hid', type=int, default=300)
    parser.add_argument('--seq_w_layer', type=int, default=1)
    parser.add_argument('--seq_droprate', type=float, default=0.5)
    parser.add_argument('--seq_model', choices=['vanilla', 'crf'], default='vanilla')
    parser.add_argument('--seq_rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')

    parser.add_argument('--eval_type', choices=["f1", "acc"], default="f1")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='SGD')
    parser.add_argument('--rand_seed', type=int, default=999)
    parser.add_argument('--sigma', type=float, default=0)
    parser.add_argument('--noise_module', type=str, default=all, help='modules to be applied noise')
    args = parser.parse_args()

    # automatically sync to spreadsheet
    # pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking, \
    #                   sheet_track_name=args.spreadsheet_name, credential_path="/data/work/jingbo/ll2/Torch-Scope/torch-scope-8acf12bee10f.json")
    
    args.params = 'test_unit_' + str(args.seq_rnn_unit) + '_hidim_' + str(args.seq_w_hid) + '_layer_' + str(args.seq_w_layer) + '_drop_' + str(args.seq_droprate) + '_lr_' + str(args.lr) + '_lrDecay_' + str(args.lr_decay) + '_opt_' + str(args.update) + '_sigma_' + str(args.sigma) + '_mod_' + str(args.noise_module) + '_randseed_' + str(args.rand_seed)
    pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name, args.params), args.checkpoint_name, enable_git_track=args.git_tracking)
    pw.set_level('info')

    gpu_index = pw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)
    
    random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    pw.info('Loading data')

    dataset = pickle.load(open(args.corpus, 'rb'))
    name_list = ['gw_map', 'c_map', 'y_map', 'emb_array', 'train_data', 'test_data', 'dev_data']
    gw_map, c_map, y_map, emb_array, train_data, test_data, dev_data = [dataset[tup] for tup in name_list ]
    print("y map", y_map)
    inv_w_map = {v: k for k, v in gw_map.items()}
    inv_y_map = {v: k for k, v in y_map.items()}

    
    pw.info('Building models')

    SL_map = {'vanilla':Vanilla_SeqLabel, 'crf':SeqLabel_CRF}
    seq_model = SL_map[args.seq_model](len(c_map), args.seq_c_dim, args.seq_c_hid, args.seq_c_layer, len(gw_map), args.seq_w_dim, args.seq_w_hid, args.seq_w_layer, len(y_map), args.seq_droprate, unit=args.seq_rnn_unit)
    seq_model.rand_init(seed=args.rand_seed)
    seq_model.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    seq_config = seq_model.to_params()
    seq_model.to(device)
    crit = CRFLoss(y_map)
    decoder = CRFDecode(y_map)
    evaluator = eval_wc(decoder, args.eval_type, inv_w_map, inv_y_map)

    if args.restore_model_path != None:
        pw.info('Loading checkpoint for seq model from {}'.format(args.restore_model_path))
        seq_model_dict = pw.restore_best_checkpoint(args.restore_model_path)
        seq_model_dict = seq_model_dict['model']
        if args.noise_module == 'all':
            keys = seq_model_dict.keys()
        elif args.noise_module == 'crf_transition':
            keys = ['crf.transitions']
        else: # none
            keys = []
        for key in keys:
            value = seq_model_dict[key]
            sz = value.view(-1).size()
            noise = torch.torch.FloatTensor(np.random.normal(0, args.sigma, sz))
            noise = noise.reshape_as(value)
            seq_model_dict[key] = value + noise
        seq_model.load_state_dict(seq_model_dict)


    pw.info('Constructing dataset')

    dataset = pickle.load(open(args.corpus_test, 'rb'))
    name_list = ['gw_map', 'c_map', 'y_map', 'emb_array', 'train_data', 'test_data', 'dev_data']
    gw_map, c_map, y_map, emb_array, train_data, test_data, dev_data = [dataset[tup] for tup in name_list ]
    print("y map", y_map)
    inv_w_map = {v: k for k, v in gw_map.items()}
    inv_y_map = {v: k for k, v in y_map.items()}

    train_dataset, test_dataset, dev_dataset = [SeqDataset(tup_data, gw_map['<\n>'], c_map[' '], c_map['\n'], y_map['<s>'], y_map['<eof>'], len(y_map), args.batch_size, if_shuffle=False) for tup_data in [train_data, test_data, dev_data]]
    #test_dataset, dev_dataset = [SeqDataset(tup_data, gw_map['<\n>'], c_map[' '], c_map['\n'], y_map['<s>'], y_map['<eof>'], len(y_map), args.batch_size) for tup_data in [test_data, dev_data]]

    pw.info('Constructing optimizer')

    param_dict = filter(lambda t: t.requires_grad, seq_model.parameters())
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](param_dict, lr=args.lr)
    else:
        optimizer=optim_map[args.update](param_dict)

    pw.info('Saving configues.')
    pw.save_configue(args)


    path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'train')
    test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, train_dataset.get_tqdm(device), save_path=path)

    pw.info('test f1: {}'.format(test_f1))
    pw.info('test pre: {}'.format(test_pre))
    pw.info('test rec: {}'.format(test_rec))
      
    pw.close()
