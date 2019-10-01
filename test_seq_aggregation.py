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
from model_seq.evaluator_with_attention import eval_wc
from model_seq.seqlabel_aggregation import Vanilla_SeqLabel
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
import traceback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='teacher-networks', help='maAddVecCrowd,maCatVecCrowd,maMulVecCrowd,maMulMatCrowd, maMulCRFCrowd, maMulScoreCrowd')

    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint')
    parser.add_argument('--checkpoint_name', default='ner')
    parser.add_argument('--git_tracking', action='store_true')

    parser.add_argument('--restore_model_path', default=None)

    parser.add_argument('--corpus', default='./data/ner_dataset.pk')

    parser.add_argument('--seq_c_dim', type=int, default=30)
    parser.add_argument('--seq_c_hid', type=int, default=150)
    parser.add_argument('--seq_c_layer', type=int, default=1)
    parser.add_argument('--seq_w_dim', type=int, default=100)
    parser.add_argument('--seq_w_hid', type=int, default=300)
    parser.add_argument('--seq_w_layer', type=int, default=1)
    parser.add_argument('--seq_droprate', type=float, default=0.5)
    parser.add_argument('--seq_model', choices=['vanilla','vanilla_crowd'], default='vanilla')
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
    parser.add_argument('--annotator_num', type=int, default=3)
    parser.add_argument('--model_require_update', type=str, default='True')
    parser.add_argument('--snt_emb', type=str, default='mean')

    args = parser.parse_args()

    # automatically sync to spreadsheet
    # pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking, \
    #                   sheet_track_name=args.spreadsheet_name, credential_path="/data/work/jingbo/ll2/Torch-Scope/torch-scope-8acf12bee10f.json")
    tasks = ['latent','maAddVecCrowd','maCatVecCrowd','maMulVecCrowd','maMulMatCrowd', 'maMulCRFCrowd', 'maMulScoreCrowd']
    for subtask in args.task.split('+'):
        assert(subtask in tasks)
    
    args.params = 'test_matrix_' + str(args.snt_emb) + '_unit_' + str(args.seq_rnn_unit) + '_hidim_' + str(args.seq_w_hid) + '_layer_' + str(args.seq_w_layer) + '_drop_' + str(args.seq_droprate) + '_lr_' + str(args.lr) + '_lrDecay_' + str(args.lr_decay) + '_opt_' + str(args.update) + '_batchSize_' + str(args.batch_size) + '_modelUpdate_' + str(args.model_require_update) + '_randseed_' + str(args.rand_seed) + '_batchSize_' + str(args.batch_size)
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
    inv_w_map = {v: k for k, v in gw_map.items()}
    inv_y_map = {v: k for k, v in y_map.items()}

    
    pw.info('Building models')

    SL_map = {'student':Vanilla_SeqLabel}
    seq_model = SL_map['student'](len(c_map), args.seq_c_dim, args.seq_c_hid, args.seq_c_layer, len(gw_map), args.seq_w_dim, args.seq_w_hid, args.seq_w_layer, len(y_map), args.seq_droprate, args.annotator_num, unit=args.seq_rnn_unit, task=args.task, snt_emb=args.snt_emb)
    seq_model.rand_init()
    seq_model.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    seq_config = seq_model.to_params()
    seq_model.to(device)
    crit = CRFLoss(y_map)
    decoder = CRFDecode(y_map)
    evaluator = eval_wc(decoder, args.eval_type, inv_w_map, inv_y_map)
    models = [seq_model]

    if args.restore_model_path != None:
        pw.info('Loading checkpoint for seq model from {}'.format(args.restore_model_path))
        seq_model_dict = pw.restore_best_checkpoint(args.restore_model_path)
        pw.info('seq model dict:', seq_model_dict)
        seq_model.load_state_dict(seq_model_dict['model'])

    pw.info('Constructing dataset')

    test_dataset, dev_dataset = [SeqDataset(tup_data, gw_map['<\n>'], c_map[' '], c_map['\n'], y_map['<s>'], y_map['<eof>'], len(y_map), args.batch_size, if_shuffle=False) for tup_data in [test_data, dev_data]]

    pw.info('Constructing optimizer')

    
    #param_dict = filter(lambda t: t.requires_grad, seq_model.parameters())
    #params = list(seq_model.parameters()) + list(crit.parameters())
    params = list(seq_model.parameters())
    
    param_dict = filter(lambda t: t.requires_grad, params)
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](param_dict, lr=args.lr)
    else:
        optimizer=optim_map[args.update](param_dict)

    pw.info('Saving configues.')
    pw.save_configue(args)

    pw.info('checking model parameters')
    print(param_dict)
    for name, param in seq_model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    num_params = sum(p.numel() for p in seq_model.parameters() if p.requires_grad)
    print("Model num para#:", num_params)

    pw.info('Setting up training environ.')
    best_f1 = float('-inf')
    patience_count = 0
    batch_index = 0
    normalizer=0
    tot_loss = 0

    print('seq_model.crf.attention',seq_model.crf.attention.data)
    path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'test_'+str(0)+'th')
    test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device), save_path=path)
    pw.info('Initial model performance:')
    pw.info('test f1: {}'.format(test_f1))
    pw.close()

