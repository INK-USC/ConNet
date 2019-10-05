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
from model.models import *
from model.crf import *
from model.predictor import *
from model.evaluator import *
from model.trainer import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('-data_dir', help='path to everything')
    p.add_argument('-data_file', help='data file name')
    p.add_argument('-evaluator', choices=['chunk', 'token'], help='which evaluator to use')
    
    # arguments for RNN model
    p.add_argument('-char_emb', type=str, default=None)
    p.add_argument('-char_rnn', action='store_true')
    p.add_argument('-char_rnn_hid', type=int, default=30)
    p.add_argument('-char_rnn_layers', type=int, default=1)
    p.add_argument('-word_rnn', action='store_true')
    p.add_argument('-word_rnn_hid', type=int, default=100)
    p.add_argument('-word_rnn_layers', type=int, default=1)
    p.add_argument('-word_emb', type=str, default=None)
    p.add_argument('-fine_tune_char_emb', action='store_true')
    p.add_argument('-fine_tune_word_emb', action='store_true')
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
    p.add_argument('-model', type=str, choices=['STM', 'MTM', 'MTM_bea', 'CN_2', 'CN_2_with_agg', 'CN_2_with_bea', 'CN_2_alternate', 'FCN_2', 'FCN_2_alternate', 'FCN_2_v2', \
                                                'FCN_2_alternate_v2', 'ADV_CN_2', 'ADV_FCN_v2', 'Peng_2016_mask', 'Peng_2016_trans', 'Crowd_add', 'Crowd_cat', 'gold'], help='which model to use')
    p.add_argument('-cm_dim', type=int, help='dimmension of the crowd layer')
    p.add_argument('-fine_tune', type=int, choices=[0,1], help='(for consensus network) whether to fine tune LSTM-CRF in step 2')
    p.add_argument('-adv_lr', type=float, default=0.0003, help='(for adverserial consensus network) lr for adverserial training')
    p.add_argument('-mode', type=str, choices=['supervised', 'leave-1-out', 'low-resource'], default='supervised', help='training mode')
    p.add_argument('-target_task', type=str, default=None, help="(for 'leave-1-out' and 'low-resource' settings) the target task")
    p.add_argument('-down_sample', type=int, default=None, help="(for 'low-resource' setting) down sample the target task")
    
    
    args = p.parse_args()
    
    print(args)
    
    # For debugging
    # pickle.dump(args, open('pickle_breakpoints/train_args.p', 'wb'))
    # assert False
    # args = pickle.load(open('pickle_breakpoints/train_args.p', 'rb'))
    
    args = vars(args)
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    # load data
    common = pickle.load(open(args['data_dir']+'/common.p', 'rb'))
    label2idx = common['label2idx']
    task2idx = common['task2idx']
    data = pickle.load(open(args['data_dir']+'/'+args['data_file'], 'rb'))
    train_data = data['train_data']
    dev_data = data['dev_data']
    test_data = data['test_data']
    train_data = [r['feats'] for r in train_data]
    dev_data = [r['feats'] for r in dev_data]
    test_data = [r['feats'] for r in test_data]
    
    
    if 'panx' in args['data_dir']:
        for i, d in enumerate(train_data):
            d['data_id'] = i
    
    
    # Test: down-sampling
    # print('# sents for the target task before down-sampling:', len([r['data_id'] for r in train_data if r['task'] == task2idx[args['target_task']]]))
    # target_task_train = [r['data_id'] for r in train_data if r['task'] == task2idx[args['target_task']]]
    # selected = np.random.choice(target_task_train, args['down_sample'], replace=False)
    # train_data = [r for r in train_data if r['task']!=task2idx[args['target_task']] or r['data_id'] in selected]
    # print('# sents for the target task after down-sampling:', len([r['data_id'] for r in train_data if r['task'] == task2idx[args['target_task']]]))
    
    emb = [None, None]
    if args['word_rnn']:
        word = pickle.load(open(args['data_dir']+'/'+args['word_emb'], 'rb'))
        word2idx = word['word2idx']
        word_emb = word['word_emb']
        for data_ in [train_data, dev_data, test_data]:
            for d in data_:
                d['words'] = [word2idx[r.lower()] if r.lower() in word2idx else word2idx['<unk>'] for r in d['words']]
        emb[1] = word_emb
        args['pad_word_idx'] = word2idx['<pad>']
    
    if args['char_rnn']:
        char = pickle.load(open(args['data_dir']+'/'+args['char_emb'], 'rb'))
        char2idx = char['char2idx']
        char_emb = char['char_emb']
        emb[0] = char_emb
        args['pad_char_idx'] = char2idx['<pad>']
    
    random.shuffle(train_data)
    
    args['label2idx'] = label2idx
    args['tagset_size'] = len(label2idx)
    # args['num_tasks'] = len(task2idx) if args['mode'] != 'leave-1-out' else len(task2idx)-1
    args['num_tasks'] = len(task2idx)
    args['pad_label_idx'] = label2idx['<pad>']
    args['start_label_idx'] = label2idx['<start>']
    
    decoder = CRFDecode_vb(len(label2idx), label2idx['<start>'], label2idx['<pad>'])
    criterion = CRFLoss_vb(len(label2idx), label2idx['<start>'], label2idx['<pad>'])
    predictor = Predictor_CRF(args, decoder, task2idx)
    if args['evaluator'] == 'token':
        evaluator = Token_Evaluator(args, predictor)
        trainer = Trainer(args, evaluator, criterion, task2idx, 'acc')
    else:
        evaluator = Chunk_Evaluator(args, predictor, label2idx)
        trainer = Trainer(args, evaluator, criterion, task2idx, 'f1')
    
    
    if args['model'] == 'STM':
        best_model = STM(args, emb) # make another model to save the best state_dict, because deepcopying a model causes issue for LSTM
        model_ = STM(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_stm(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            args['epochs'] = [args['epochs'], args['epochs']]
            args['epochs'] = trainer.train_stm_low(train_data, dev_data, test_data, model_, best_model, 1, task2idx[args['target_task']])
        # model_ = STM(args, emb)
        # torch.cuda.empty_cache()
        # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        # if args['mode'] in ['supervised', 'leave-1-out']:
            # trainer.train_stm(train_data+dev_data, dev_data, test_data, model_, best_model, 2, args['mode'], task2idx[args['target_task']])
        # else:
            # trainer.train_stm_low(train_data+dev_data, dev_data, test_data, model_, best_model, 2, task2idx[args['target_task']])
    
    elif args['model'] == 'gold':
        best_model = STM(args, emb)
        model_ = STM(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_gold(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'Crowd_add':
        best_model = Crowd_add(args, emb)
        model_ = Crowd_add(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_crowd_add(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'Crowd_cat':
        best_model = Crowd_cat(args, emb)
        model_ = Crowd_cat(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_crowd_add(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'CN_2':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_2(args, emb)
        model_ = CN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_cn2(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            args['epochs'] = trainer.train_cn2_low(train_data, dev_data, test_data, model_, best_model, 1, task2idx[args['target_task']])
        
        # model_ = CN_2(args, emb)
        # torch.cuda.empty_cache()
        # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        # if args['mode'] in ['supervised', 'leave-1-out']:
            # trainer.train_cn2(train_data+dev_data, None, test_data, model_, best_model, 2, args['mode'], task2idx[args['target_task']])
        # else:
            # trainer.train_cn2_low(train_data+dev_data, dev_data, test_data, model_, best_model, 2, task2idx[args['target_task']])
    
    elif args['model'] == 'CN_2_with_agg':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_2(args, emb)
        model_ = CN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_with_agg(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
    
    elif args['model'] == 'CN_2_with_bea':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_2(args, emb)
        model_ = CN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_with_agg(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']], agg='BEA')

    elif args['model'] == 'CN_2_alternate':
        best_model = CN_2(args, emb)
        model_ = CN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_alternate(train_data, dev_data, test_data, model_, best_model, exp=1)
        
        model_ = CN_2(args, emb)
        torch.cuda.empty_cache()
        print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        trainer.train_cn2_alternate(train_data+dev_data, None, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'FCN_2':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = FCN_2(args, emb)
        model_ = FCN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2(train_data, dev_data, test_data, model_, best_model, exp=1)
        
        model_ = FCN_2(args, emb)
        torch.cuda.empty_cache()
        print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        trainer.train_cn2(train_data+dev_data, None, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'FCN_2_alternate':
        best_model = FCN_2(args, emb)
        model_ = FCN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_alternate(train_data, dev_data, test_data, model_, best_model, exp=1)
        
        model_ = FCN_2(args, emb)
        torch.cuda.empty_cache()
        print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        trainer.train_cn2_alternate(train_data+dev_data, None, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'FCN_2_v2':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = FCN_2_v2(args, emb)
        model_ = FCN_2_v2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        if args['mode'] in ['supervised', 'leave-1-out']:
            args['epochs'] = trainer.train_cn2(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        else:
            args['epochs'] = trainer.train_cn2_low(train_data, dev_data, test_data, model_, best_model, 1, task2idx[args['target_task']])
        
        # model_ = FCN_2_v2(args, emb)
        # torch.cuda.empty_cache()
        # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        # if args['mode'] in ['supervised', 'leave-1-out']:
            # trainer.train_cn2(train_data+dev_data, None, test_data, model_, best_model, 2, args['mode'], task2idx[args['target_task']])
        # else:
            # trainer.train_cn2_low(train_data+dev_data, dev_data, test_data, model_, best_model, 2, task2idx[args['target_task']])
    
    elif args['model'] == 'FCN_2_alternate_v2':
        best_model = FCN_2_v2(args, emb)
        model_ = FCN_2_v2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_alternate(train_data, dev_data, test_data, model_, best_model, exp=1)
        
        model_ = FCN_2_v2(args, emb)
        torch.cuda.empty_cache()
        print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        trainer.train_cn2_alternate(train_data+dev_data, None, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'MTM':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_2(args, emb)
        model_ = CN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_mtm(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']])
        
        # model_ = MTM(args, emb)
        # torch.cuda.empty_cache()
        # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        # trainer.train_mtm(train_data, dev_data, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'MTM_bea':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = CN_2(args, emb)
        model_ = CN_2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_mtm(train_data, dev_data, test_data, model_, best_model, 1, args['mode'], task2idx[args['target_task']], agg='BEA')
        
        # model_ = MTM(args, emb)
        # torch.cuda.empty_cache()
        # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        # trainer.train_mtm(train_data, dev_data, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'ADV_CN_2':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = ADV_CN(args, emb)
        model_ = ADV_CN(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_adv(train_data, dev_data, test_data, model_, best_model, exp=1)
        
        model_ = ADV_CN(args, emb)
        torch.cuda.empty_cache()
        print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        trainer.train_cn2_adv(train_data+dev_data, None, test_data, model_, best_model, exp=2)
    
    elif args['model'] == 'ADV_FCN_v2':
        args['epochs'] = [args['epochs'], args['epochs']]
        best_model = ADV_FCN_v2(args, emb)
        model_ = ADV_FCN_v2(args, emb)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = trainer.train_cn2_adv(train_data, dev_data, test_data, model_, best_model, exp=1)
        
        model_ = ADV_FCN_v2(args, emb)
        torch.cuda.empty_cache()
        print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        trainer.train_cn2_adv(train_data+dev_data, None, test_data, model_, best_model, exp=2)
    
    elif args['model'] in ['Peng_2016_mask', 'Peng_2016_trans']:
        # currently only works in low-resource setting
        method = 'domain_mask' if args['model'] == 'Peng_2016_mask' else 'domain_trans'
        best_model = Peng_2016(args, emb, method)
        model_ = Peng_2016(args, emb, method)
        print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
        args['epochs'] = [args['epochs'], args['epochs']]
        args['epochs'] = trainer.train_Peng_2016_low(train_data, dev_data, test_data, model_, best_model, 1, task2idx[args['target_task']])
        
        # model_ = Peng_2016(args, emb, method)
        # torch.cuda.empty_cache()
        # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
        # trainer.train_Peng_2016_low(train_data+dev_data, dev_data, test_data, model_, best_model, 2, task2idx[args['target_task']])
