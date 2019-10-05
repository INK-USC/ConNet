from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_seq.crf_extraction import CRFLoss, CRFDecode, CRFLoss_ma 
from model_seq.dataset import SeqDataset
from model_seq.dataset_ma import SeqDatasetMA
from model_seq.evaluator_extraction import eval_wc, eval_wc_latent, eval_wc_trainset
from model_seq.seqlabel_extraction import Vanilla_SeqLabel
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

    parser.add_argument('--task', type=str, default='teacher-networks', help='maAddCrowd,maCatCrowd,maMulCrowd,maMulMatCrowd, maMulCRFCrowd, maMulScoreCrowd, maMulCRFScoreCrowd')

    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='/data/ouyu/exp/Vanilla_NER/checkpoint')
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
    parser.add_argument('--selecting', choices=['avg','latent'], default='avg')

    args = parser.parse_args()

    # automatically sync to spreadsheet
    # pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking, \
    #                   sheet_track_name=args.spreadsheet_name, credential_path="/data/work/jingbo/ll2/Torch-Scope/torch-scope-8acf12bee10f.json")
    if args.selecting == 'latent':
        assert('latent' in args.task)
    
    args.params = 'test_unit_' + str(args.seq_rnn_unit) + '_hidim_' + str(args.seq_w_hid) + '_layer_' + str(args.seq_w_layer) + '_drop_' + str(args.seq_droprate) + '_lr_' + str(args.lr) + '_lrDecay_' + str(args.lr_decay) + '_opt_' + str(args.update) + '_select_' + str(args.selecting) + '_epoch_'+ str(args.epoch) +'_batchSize_' + str(args.batch_size) + '_randseed_' + str(args.rand_seed)
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

    print(len(gw_map))
    args.annotator_num = len(train_data[0][-1][0])
    pw.info("Annotator num: {}".format(args.annotator_num))

    
    pw.info('Building models')

    SL_map = {'vanilla':Vanilla_SeqLabel}
    seq_model = SL_map[args.seq_model](len(c_map), args.seq_c_dim, args.seq_c_hid, args.seq_c_layer, len(gw_map), args.seq_w_dim, args.seq_w_hid, args.seq_w_layer, len(y_map), args.seq_droprate, args.annotator_num, unit=args.seq_rnn_unit, task=args.task)
    seq_model.rand_init()
    seq_model.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    seq_config = seq_model.to_params()
    seq_model.to(device)

    if args.restore_model_path != None:
        pw.info('Loading checkpoint for seq model from {}'.format(args.restore_model_path))
        seq_model_dict = pw.restore_best_checkpoint(args.restore_model_path)
        seq_model.load_state_dict(seq_model_dict['model'])
        crowd = seq_model_dict['model']['crf.maMulScoreCrowd']
        crowd_dist = np.zeros((args.annotator_num, args.annotator_num))
        for i in range(args.annotator_num):
            for j in range(args.annotator_num):
                crowd_dist[i][j] = np.linalg.norm(crowd[i] - crowd[j])
        print(crowd_dist)
        pw.info('crowd distance matrix:')
        pw.info(crowd_dist)

    crit = CRFLoss_ma(y_map, task=args.task, a_num=args.annotator_num)
    decoder = CRFDecode(y_map)
    evaluator = eval_wc(decoder, args.eval_type, args.annotator_num, inv_w_map, inv_y_map)
    latent_evaluator = eval_wc_latent(decoder, args.eval_type, inv_w_map, inv_y_map)
    trainset_evaluator = eval_wc_trainset(decoder, args.eval_type, args.annotator_num, inv_w_map, inv_y_map)
    models = [seq_model]

    pw.info('Constructing dataset')

    train_dataset = SeqDatasetMA(train_data, gw_map['<\n>'], c_map[' '], c_map['\n'], y_map['<s>'], y_map['<eof>'], len(y_map), args.batch_size, y_unk=y_map['<unk>'], y_O=y_map['O'], if_shuffle=False)
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

    pw.info('Setting up training environ.')
    best_f1 = float('-inf')
    patience_count = 0
    batch_index = 0
    normalizer=0
    tot_loss = 0

    # train
    path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'train')
    scores = trainset_evaluator.calc_score(seq_model, train_dataset.get_tqdm(device), save_path=path, allset=False)
    scores = np.array(scores).T
    dev_f1_list, dev_pre_list, dev_rec_list, dev_acc_list = scores
    pw.info('train f1: {}'.format(dev_f1_list))
    pw.info('avg train f1: {}'.format(np.mean(dev_f1_list)))

    path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'train_allset')
    scores = trainset_evaluator.calc_score(seq_model, train_dataset.get_tqdm(device), save_path=path, allset=True)
    scores = np.array(scores).T
    dev_f1_list, dev_pre_list, dev_rec_list, dev_acc_list = scores
    pw.info('train f1: {}'.format(dev_f1_list))
    pw.info('avg train f1: {}'.format(np.mean(dev_f1_list)))

    # dev
    path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'dev')
    scores = evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device), save_path=path)
    scores = np.array(scores).T 
    dev_f1_list, dev_pre_list, dev_rec_list, dev_acc_list = scores 
    pw.info('dev f1: {}'.format(dev_f1_list))
    pw.info('avg dev f1: {}'.format(np.mean(dev_f1_list)))
    if 'latent' in args.task:
        dev_f1, dev_pre, dev_rec, dev_acc = latent_evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device))
        pw.info('latent dev f1: {}'.format(dev_f1))
        pw.info('dev pre: {}'.format(dev_pre))
        pw.info('dev rec: {}'.format(dev_rec))

    # test
    path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'test')
    scores = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device), save_path=path)
    scores = np.array(scores).T 
    test_f1_list, test_pre_list, test_rec_list, test_acc_list = scores 
    pw.info('test f1: {}'.format(test_f1_list))
    pw.info('avg test f1: {}'.format(np.mean(test_f1_list)))
    if 'latent' in args.task:
        path = os.path.join(args.cp_root, args.checkpoint_name, args.params, 'test_latent')
        test_f1, test_pre, test_rec, test_acc = latent_evaluator.calc_score(seq_model, test_dataset.get_tqdm(device), save_path=path)
        pw.info('latent test f1: {}'.format(test_f1))
        pw.info('test pre: {}'.format(test_pre))
        pw.info('test rec: {}'.format(test_rec))

    pw.close()
