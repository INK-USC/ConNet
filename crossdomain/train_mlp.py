from pathlib import Path
import pickle
import sys, os
import argparse
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import random
import time
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from model.models import *
from model.crf import *
from model.predictor import *
from model.evaluator import *
from model.trainer import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('-data_dir', help='path to everything')
    p.add_argument('-data_file', help='data file name')
    p.add_argument('-evaluator', choices=['binary'], help='which evaluator to use')
    
    # arguments for RNN model
    p.add_argument('-mlp', action='store_true')
    p.add_argument('-mlp_hid', type=int, default=100)
    p.add_argument('-batch', type=int, default=64, help='batch size')
    p.add_argument('-epochs', type=int, default=200)
    p.add_argument('-patience', type=int, default=30)
    p.add_argument('-seed', type=int, default=999)
    p.add_argument('-clip_grad', type=float, default=5)
    p.add_argument('-lr', type=float, default=0.015)
    p.add_argument('-lr_decay', type=float, default=0.05)
    p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-cuda', action='store_true')
    p.add_argument('-save_model', type=str, help='path to save the model checkpoint')
    p.add_argument('-model', type=str, choices=['STM_MLP', 'MTM_MLP', 'MTM_MLP_bea', 'CN_MLP', 'CN_MLP_with_agg', 'CN_MLP_with_bea', \
                                                'Crowd_add_MLP', 'Crowd_cat_MLP', 'gold_MLP'], help='which model to use')
    p.add_argument('-cm_dim', type=int, help='dimmension of the crowd layer')
    p.add_argument('-fine_tune', type=int, choices=[0,1], help='(for consensus network) whether to fine tune LSTM-CRF in step 2')
    p.add_argument('-mode', type=str, choices=['supervised', 'leave-1-out', 'low-resource'], default='supervised', help='training mode')
    p.add_argument('-target_task', type=str, default=None, help="(for 'leave-1-out' and 'low-resource' settings) the target task")
    p.add_argument('-down_sample', type=int, default=None, help="(for 'low-resource' setting) down sample the target task")
    
    
    args = p.parse_args()
    
    print(args)
    
    # For debugging
    # pickle.dump(args, open('pickle_breakpoints/train_mlp_args.p', 'wb'))
    # assert False
    # args = pickle.load(open('pickle_breakpoints/train_mlp_args.p', 'rb'))
    
    args = vars(args)
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    # load data
    common = pickle.load(open(args['data_dir']+'/common.p', 'rb'))
    task2idx = common['task2idx']
    data = pickle.load(open(args['data_dir']+'/'+args['data_file'], 'rb'))
    data = data['data']
    test_data = [r for r in data if r['ref']['task']==args['target_task']]
    train_dev = [r for r in data if r['ref']['task']!=args['target_task']]
    train_data, dev_data = train_test_split(train_dev, test_size=0.33, random_state=0, stratify=[r['feats']['task'] for r in train_dev])
    train_data = [r['feats'] for r in train_data]
    dev_data = [r['feats'] for r in dev_data]
    test_data = [r['feats'] for r in test_data]
    
    #sanity check
    # for d in test_data:
        # d['label'] = 0
    
    # Test: down-sampling
    # print('# sents for the target task before down-sampling:', len([r['data_id'] for r in train_data if r['task'] == task2idx[args['target_task']]]))
    # target_task_train = [r['data_id'] for r in train_data if r['task'] == task2idx[args['target_task']]]
    # selected = np.random.choice(target_task_train, args['down_sample'], replace=False)
    # train_data = [r for r in train_data if r['task']!=task2idx[args['target_task']] or r['data_id'] in selected]
    # print('# sents for the target task after down-sampling:', len([r['data_id'] for r in train_data if r['task'] == task2idx[args['target_task']]]))
    
    
    random.shuffle(train_data)
    
    args['num_tasks'] = len(task2idx)
    
    criterion = nn.BCELoss()
    predictor = Predictor_Binary(args)
    if args['evaluator'] == 'binary':
        evaluator = Binary_Evaluator(args, predictor)
        trainer = Trainer(args, evaluator, criterion, task2idx, 'acc')
    else:
        pass
    
    
    if args['model'] == 'STM_MLP':
        best_model = STM_MLP(args) # make another model to save the best state_dict, because deepcopying a model causes issue for LSTM
        model_ = STM_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_stm_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            assert False
    
    if args['model'] == 'gold_MLP':
        best_model = STM_MLP(args)
        model_ = STM_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_gold_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'Crowd_add_MLP':
        best_model = Crowd_add_MLP(args)
        model_ = Crowd_add_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_crowd_add_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'Crowd_cat_MLP':
        best_model = Crowd_cat_MLP(args)
        model_ = Crowd_cat_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_crowd_add_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'CN_MLP':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_MLP(args)
        model_ = CN_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_cn_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            assert False
    
    elif args['model'] == 'CN_MLP_with_agg':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_MLP(args)
        model_ = CN_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_cn_mlp_with_agg(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            assert False
    
    elif args['model'] == 'CN_MLP_with_bea':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_MLP(args)
        model_ = CN_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_cn_mlp_with_agg(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']], agg='BEA')
        else:
            assert False
    
    elif args['model'] == 'MTM_MLP':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_MLP(args)
        model_ = CN_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_mtm_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            assert False
    
    elif args['model'] == 'MTM_MLP_bea':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_MLP(args)
        model_ = CN_MLP(args)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_mtm_mlp(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']], agg='BEA')
        else:
            assert False
        