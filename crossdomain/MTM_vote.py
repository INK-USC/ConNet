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
    
    p.add_argument('-checkpoint', help='checkpoint file')
    
    args = p.parse_args()
    
    print(args)
    
    # For debugging
    # pickle.dump(args, open('pickle_breakpoints/vote_args.p', 'wb'))
    # assert False
    # args = pickle.load(open('pickle_breakpoints/vote_args.p', 'rb'))
    
    args = vars(args)
    
    checkpoint_file = os.listdir(args['checkpoint'])
    assert len(checkpoint_file) == 1
    checkpoint = torch.load(args['checkpoint'] + '/' + checkpoint_file[0])
    
    args = checkpoint['args']

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    # load data
    common = pickle.load(open(args['data_dir']+'/common.p', 'rb'))
    label2idx = common['label2idx']
    task2idx = common['task2idx']
    data = pickle.load(open(args['data_dir']+'/data.p', 'rb'))
    test_data = data['test_data']
    test_data = [r['feats'] for r in test_data]
    
    model_ = MTM(args, \
                 [checkpoint['state_dict']['char_emb.weight'].cpu().numpy() if args['char_rnn'] else None, \
                  checkpoint['state_dict']['word_emb.weight'].cpu().numpy()])
    model_.load_state_dict(checkpoint['state_dict'])
    target_task = args['target_task']
    if args['cuda']:
        model_ = model_.cuda()
    
    decoder = CRFDecode_vb(len(label2idx), label2idx['<start>'], label2idx['<pad>'])
    predictor = Predictor(args, decoder, task2idx)
    criterion = CRFLoss_vb(len(label2idx), label2idx['<start>'], label2idx['<pad>'])
    if args['evaluator'] == 'token':
        evaluator = Token_Evaluator(args, predictor)
        eval_metric = 'acc'
    else:
        evaluator = Chunk_Evaluator(args, predictor, label2idx)
        eval_metric = 'f1'
    
    test_data = [r for r in test_data if r['task']==task2idx[target_task]]
    word = pickle.load(open(args['data_dir']+'/'+args['word_emb'], 'rb'))
    word2idx = word['word2idx']
    word_emb = word['word_emb']
    for d in test_data:
        d['words'] = [word2idx[r.lower()] if r.lower() in word2idx else word2idx['<unk>'] for r in d['words']]

    preds = []
    for task_name, task_id in task2idx.items():
        if task_name == target_task:
            continue
        
        # evaluator.evaluate(test_data, model_, criterion, 'mtm', task_id=task_id)
        preds.append(predictor.predict(test_data, model_, criterion, 'mtm', task_id=task_id)[0])
    
    vote = []
    for i in range(len(preds[0])):
        curr_pred = [r[i] for r in preds]
        curr_vote = []
        for j in range(len(curr_pred[0])):
            curr_curr_pred = [r[j] for r in curr_pred]
            curr_vote.append(max(set(curr_curr_pred), key=curr_curr_pred.count))
        
        vote.append(curr_vote)
    
    acc, prec, recall, f1, loss = evaluator.evaluate(test_data, model_, criterion, 'mtm', preds=vote)
    curr_score = acc if eval_metric=='acc' else f1
    
    print("Target task:", target_task)
    print("Performance on target task:")
    print("A: %.4f  P: %.4f  R: %.4f  F: %.4f\n" % (acc, prec, recall, f1))