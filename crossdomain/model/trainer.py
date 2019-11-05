import numpy as np
import time, copy
import torch
import torch.optim as optim
import torch.nn as nn
from model.data_utils import *
from model.utils import *
from model.bea_aggregate import bea_ent_x2_t10, bea_ent_uns


class Trainer():
    def __init__(self, args, evaluator, criterion, task2idx, metric):
        self.args = args
        self.evaluator = evaluator
        self.criterion = criterion
        self.task2idx = task2idx
        self.metric = metric
    
    def train_epoch(self, train_data, model_, optimizer, model_name):
        model_.train()
        if model_name == 'Peng2016':
            train_data = pack_data_tasks(self.args, train_data, self.task2idx)
        elif model_name in ['stm_mlp', 'cn_mlp_extraction', 'cn_mlp_aggregation', 'mtm_mlp', 'crowd_add_mlp', 'crowd_add_mlp_test']:
            train_data = pack_data_mlp(self.args, train_data)
        else:
            train_data = pack_data(self.args, train_data)
        for batch in train_data:
            model_.zero_grad()
            if model_name == 'stm':
                crf_out = model_(batch['data_input'])
            elif model_name == 'stm_mlp':
                crf_out = model_(batch['data_input'])
            elif model_name == 'mtm':
                crf_out = model_(batch['data_input'], batch['task_ids'])
            elif model_name == 'crowd_add':
                crf_out = model_(batch['data_input'], batch['task_ids'])
            elif model_name == 'crowd_add_test':
                crf_out = model_(batch['data_input'], None)
            elif model_name == 'crowd_add_mlp':
                crf_out = model_(batch['data_input'], batch['task_ids'])
            elif model_name == 'crowd_add_mlp_test':
                crf_out = model_(batch['data_input'], None)
            elif model_name == 'cn2_extraction':
                crf_out = model_(batch['data_input'], None, batch['task_ids'])
            elif model_name == 'cn2_aggregation':
                crf_out = model_(batch['data_input'], batch['sent_end'], None)
            elif model_name == 'cn_mlp_extraction':
                crf_out = model_(batch['data_input'], batch['task_ids'])
            elif model_name == 'cn_mlp_aggregation':
                crf_out = model_(batch['data_input'], None)
            elif model_name == 'adv_cn-ext':
                crf_out = model_(batch['data_input'], None, batch['task_ids'], model_name)
            elif model_name == 'adv_cn-agg':
                crf_out = model_(batch['data_input'], batch['sent_end'], None, model_name)
            elif model_name in ['adv_cn-train_cla', 'adv_cn-adv']:
                cla_out = model_(batch['data_input'], batch['sent_end'], None, model_name)
            elif model_name == 'Peng2016':
                crf_out = model_(batch['data_input'], batch['task_ids'][0])
            else:
                assert False
            
            if model_name == 'adv_cn-train_cla':
                task_ids = torch.tensor(batch['task_ids'])
                if self.args['cuda']:
                    task_ids = task_ids.cuda()
                loss = nn.CrossEntropyLoss()(cla_out, task_ids)
                loss.backward()
                nn.utils.clip_grad_norm_(model_.parameters(), self.args['clip_grad'])
                optimizer.step()
            elif model_name == 'adv_cn-adv':
                task_ids = torch.tensor(batch['task_ids'])
                if self.args['cuda']:
                    task_ids = task_ids.cuda()
                loss = -nn.CrossEntropyLoss()(cla_out, task_ids)
                loss.backward()
                nn.utils.clip_grad_norm_(model_.parameters(), self.args['clip_grad'])
                optimizer.step()
            elif model_name in ['stm_mlp', 'cn_mlp_extraction', 'cn_mlp_aggregation', 'mtm_mlp', 'crowd_add_mlp', 'crowd_add_mlp_test']:
                loss = self.criterion(crf_out, batch['label'])
                loss.backward()
                nn.utils.clip_grad_norm_(model_.parameters(), self.args['clip_grad'])
                optimizer.step()
            else:
                crf_out = crf_out.transpose(0,1)
                mask = batch['mask'].transpose(0,1)
                label = batch['label'].transpose(0,1)
                loss = self.criterion(crf_out, label, mask)
                loss = loss.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model_.parameters(), self.args['clip_grad'])
                optimizer.step()
    
    # evaluation on one task (data contains only one task)
    def eval_task(self, data, model_, model_name):
        acc, prec, recall, f1, loss = self.evaluator.evaluate(data, model_, self.criterion, model_name)
        print("Performance on target task:")
        print("A: %.4f  P: %.4f  R: %.4f  F: %.4f\n" % (acc, prec, recall, f1))
        
        curr_score = acc if self.metric=='acc' else f1
        
        return curr_score
    
    # rough evaluation on all tasks in each epoch, only prints macro score
    def eval_epoch(self, dev_data, model_, model_name):
        curr_scores = []
        for task, task_id in self.task2idx.items():
            dev_d_t = [r for r in dev_data if r['task']==task_id]
            if not dev_d_t:
                continue
            dev_acc, dev_prec, dev_recall, dev_f1, dev_loss = self.evaluator.evaluate(dev_d_t, model_, self.criterion, model_name)
            curr_scores.append([dev_acc, dev_prec, dev_recall, dev_f1])
        
        acc, macro_prec, macro_recall, macro_f1 = [np.mean([r[i] for r in curr_scores]) for i in range(len(curr_scores[0]))]
        print("Macro scores:")
        print("A: %.4f  P: %.4f  R: %.4f  F: %.4f\n" % (acc, macro_prec, macro_recall, macro_f1))
        
        curr_score = acc if self.metric=='acc' else macro_f1
        
        return curr_score
    
    # rough evaluation on all tasks before label aggregation, returns performance for each task
    def eval_each_task(self, dev_data, model_, model_name):
        curr_scores = {}
        for task, task_id in self.task2idx.items():
            dev_d_t = [r for r in dev_data if r['task']==task_id]
            if not dev_d_t:
                continue
            dev_acc, dev_prec, dev_recall, dev_f1, dev_loss = self.evaluator.evaluate(dev_d_t, model_, self.criterion, model_name)
            curr_scores[task_id] = [task, dev_acc if self.metric=='acc' else dev_f1]
        
        print("Score for each task:")
        for t_id, (t, score) in curr_scores.items():
            print("%s:\t%.4f\n" % (t, score))
                
        return curr_scores
    
    # detailed evaluation on all tasks at last (with best model), prints score for each task
    def eval_final(self, test_data, best_model, model_name):
        scores = []
        for task, task_id in sorted(self.task2idx.items(), key=lambda x:x[0]):
            test_d_t = [r for r in test_data if r['task']==task_id]
            if not test_d_t:
                continue
            test_acc, test_prec, test_recall, test_f1, test_loss = self.evaluator.evaluate(test_d_t, best_model, self.criterion, model_name)
            scores.append([test_acc, test_prec, test_recall, test_f1])
            print("Task: %s" % task.upper())
            print("test loss: %.4f" % test_loss)
            print("A: %.4f  P: %.4f  R: %.4f  F: %.4f" % (test_acc, test_prec, test_recall, test_f1))
        
        acc, macro_prec, macro_recall, macro_f1 = [np.mean([r[i] for r in scores]) for i in range(len(scores[0]))]
        print()
        print("Macro scores:")
        print("A: %.4f  P: %.4f  R: %.4f  F: %.4f\n\n" % (acc, macro_prec, macro_recall, macro_f1))
        
        score = acc if self.metric=='acc' else macro_f1
        
        return score
    
    # train single task model in supervised and leave-1-out settings
    def train_stm(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode in ['supervised', 'leave-1-out']
        if mode == 'leave-1-out':
            train_data = [r for r in train_data if r['task']!=target_task]
            dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
            test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'stm')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'stm')
            curr_score = self.eval_epoch(dev_data, model_, 'stm')
            if exp == 1:
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'stm')
        else:
            score = self.eval_task(test_data, best_model, 'stm')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train single task model in supervised and leave-1-out settings
    def train_stm_mlp(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode == 'leave-1-out'
        train_data = [r for r in train_data if r['task']!=target_task]
        dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_.cuda()
            best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'stm_mlp')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'stm_mlp')
            curr_score = self.eval_epoch(dev_data, model_, 'stm_mlp')
            if exp == 1:
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'stm_mlp')
        else:
            score = self.eval_task(test_data, best_model, 'stm_mlp')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train single task model in low-resource setting
    def train_stm_low(self, train_data, dev_data, test_data, model_, best_model, exp, target_task=None):
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # first step: train on all tasks
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'stm')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'stm')
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'stm')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        self.eval_task(test_data, best_model, 'stm')
        
        # second step: train on target task
        train_data = [r for r in train_data if r['task']==target_task]
        dev_data = [r for r in dev_data if r['task']==target_task] if dev_data else None
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data, model_, optimizer, 'stm')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'stm')
            if exp == 1:
                curr_score = self.eval_task(dev_data, model_, 'stm')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_task(test_data, best_model, 'stm')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train single task model in leave-1-out setting with AMV label aggregation
    # *not finished
    def train_stm_with_agg(self, train_data, dev_data, test_data, model_, best_model, exp, mode='leave-1-out', target_task=None):
        assert mode == 'leave-1-out'
        if mode == 'leave-1-out':
            train_data = [r for r in train_data if r['task']!=target_task]
            dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
            test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'stm')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'stm')
            curr_score = self.eval_epoch(dev_data, model_, 'stm')
            if exp == 1:
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_task(test_data, best_model, 'stm')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train single task model in supervised and leave-1-out settings
    def train_gold(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        #train_data = [r for r in train_data if r['task']==target_task]
        dev_data = [r for r in dev_data if r['task']==target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'stm')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'stm')
            curr_score = self.eval_epoch(dev_data, model_, 'stm')
            if exp == 1:
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'stm')
        else:
            score = self.eval_task(test_data, best_model, 'stm')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train single task model in supervised and leave-1-out settings
    def train_gold_mlp(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode == 'leave-1-out'
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(test_data, test_size=0.33, random_state=0)
        if self.args['cuda']:
            model_.cuda()
            best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, 51):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'stm_mlp')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            score = self.eval_task(test_data, model_, 'stm_mlp')
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        best_model.load_state_dict(model_.state_dict())
        
        if exp == 1:
            self.save_model(best_model, 50, score, self.args['save_model']+"/Epoch"+str(50)+"_"+"%.4f"%score+'.model')
            return 50
        if exp == 2:
            self.save_model(best_model, 50, score, self.args['save_model']+"/Epoch"+str(50)+"_"+"%.4f"%score+'.model')
    
    # train single task model in supervised and leave-1-out settings
    def train_crowd_add(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode == 'leave-1-out'
        train_data = [r for r in train_data if r['task']!=target_task]
        dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'crowd_add')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'crowd_add_test')
            curr_score = self.eval_epoch(dev_data, model_, 'crowd_add')
            if exp == 1:
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_task(test_data, best_model, 'crowd_add_test')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train single task model in supervised and leave-1-out settings
    def train_crowd_add_mlp(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode == 'leave-1-out'
        train_data = [r for r in train_data if r['task']!=target_task]
        dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_score = -np.Inf
        start_time = time.time()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            self.train_epoch(train_data, model_, optimizer, 'crowd_add_mlp')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'crowd_add_mlp_test')
            curr_score = self.eval_epoch(dev_data, model_, 'crowd_add_mlp')
            if exp == 1:
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_task(test_data, best_model, 'crowd_add_mlp_test')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')

    
    
    # train consensus network (two phase) in supervised and leave-1-out settings
    def train_cn2(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode in ['supervised', 'leave-1-out']
        if mode == 'leave-1-out':
            train_data = [r for r in train_data if r['task']!=target_task]
            dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
            test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn2_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            self.eval_final(test_data, best_model, 'cn2_extraction')
        else:
            # self.eval_task(test_data, best_model, 'cn2_extraction')
            print("No evaluation method available")
        
        # Aggregation Phase
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        model_.aggregation_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data, model_, optimizer, 'cn2_aggregation')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'cn2_aggregation')
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_aggregation')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'cn2_aggregation')
        else:
            score = self.eval_task(test_data, best_model, 'cn2_aggregation')
            
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    
    # train consensus network (two phase) in supervised and leave-1-out settings
    def train_cn_mlp(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None):
        assert mode == 'leave-1-out'
        train_data = [r for r in train_data if r['task']!=target_task]
        dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_.cuda()
            best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn_mlp_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn_mlp_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            self.eval_final(test_data, best_model, 'cn_mlp_extraction')
        else:
            # self.eval_task(test_data, best_model, 'cn_mlp_extraction')
            print("No evaluation method available")
        
        # Aggregation Phase
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        model_.aggregation_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data, model_, optimizer, 'cn_mlp_aggregation')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'cn_mlp_aggregation')
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn_mlp_aggregation')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'cn_mlp_aggregation')
        else:
            score = self.eval_task(test_data, best_model, 'cn_mlp_aggregation')
            
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    # train consensus network (two phase) in supervised and leave-1-out settings
    def train_cn_mlp_with_agg(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None, agg='MV'):
        assert mode == 'leave-1-out'
        train_data_target = copy.deepcopy(test_data)
        if self.args['cuda']:
            model_.cuda()
            best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn_mlp_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn_mlp_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            self.eval_final(test_data, best_model, 'cn_mlp_extraction')
        else:
            # self.eval_task(test_data, best_model, 'cn_mlp_extraction')
            print("No evaluation method available")

        # AMV Label Aggregation
        if agg == 'MV':
            dev_scores = self.eval_each_task(dev_data, best_model, 'cn_mlp_extraction')
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(train_data_target, best_model, self.criterion, 'cn_mlp_extraction', task_id=task_id)[0]
                preds.append([dev_scores[task_id][1], curr_pred])
            
            weighted_preds = np.array([[rr * r[0] for rr in r[1]] for r in preds])
            weighted_sum = weighted_preds.sum(0)
            final_preds = weighted_sum > (weighted_preds.shape[0]/2)
            for i in range(len(train_data_target)):
                train_data_target[i]['label'] = 1 if final_preds[i] else 0
        
        elif agg == 'BEA':
            # train_data_target = test_data
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(train_data_target, best_model, self.criterion, 'cn_mlp_extraction', task_id=task_id)[0]
                preds.append(['T' if r==1 else 'F' for r in curr_pred])
            
            labels = ['T' if r['label']==1 else 'F' for r in train_data_target]
            all_labels = ['F', 'T']
            all_tags = ['F', 'T']
            preds, _ = bea_ent_uns(labels, preds, all_labels, all_tags, False)
            for i in range(len(train_data_target)):
                train_data_target[i]['label'] = 1 if preds[i]=='T' else 0
        
        else:
            assert False
        
        # Aggregation Phase
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        model_.aggregation_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data_target, model_, optimizer, 'cn_mlp_aggregation')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'cn_mlp_aggregation')
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn_mlp_aggregation')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'cn_mlp_aggregation')
        else:
            score = self.eval_task(test_data, best_model, 'cn_mlp_aggregation')
            
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    # train consensus network (two phase) in low-resource setting
    def train_cn2_low(self, train_data, dev_data, test_data, model_, best_model, exp, target_task=None):
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn2_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'cn2_extraction')
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        self.eval_task(test_data, best_model, 'cn2_extraction')
        
        # Aggregation Phase
        train_data = [r for r in train_data if r['task']==target_task]
        dev_data = [r for r in dev_data if r['task']==target_task] if dev_data else None
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        model_.aggregation_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data, model_, optimizer, 'cn2_aggregation')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'cn2_aggregation')
            if exp == 1:
                curr_score = self.eval_task(dev_data, model_, 'cn2_aggregation')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_task(test_data, best_model, 'cn2_aggregation')
            
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    # train consensus network (two phase) in leave-1-out setting with AMV label aggregation
    def train_cn2_with_agg(self, train_data, dev_data, test_data, model_, best_model, exp, mode='leave-1-out', target_task=None, agg='MV'):
        assert mode == 'leave-1-out'
        train_data_target = [r for r in train_data if r['task']==target_task]
        train_data = [r for r in train_data if r['task']!=target_task]
        dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn2_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        # AMV Label Aggregation
        if agg == 'MV':
            dev_scores = self.eval_each_task(dev_data, best_model, 'cn2_extraction')
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(train_data_target, best_model, self.criterion, 'cn2_extraction', task_id=task_id)[0]
                preds.append([dev_scores[task_id][1], curr_pred])
            
            for i in range(len(train_data_target)):
                pred_mat = np.zeros((self.args['tagset_size'], len(train_data_target[i]['label'])))
                for dev_score, curr_pred in preds:
                    curr_pred_i = curr_pred[i]
                    curr_pred_i = curr_pred_i[:len(train_data_target[i]['label'])]
                    pred_mat[curr_pred_i, range(len(curr_pred_i))] += dev_score
                    agg_pred = list(np.argmax(pred_mat, 0))
                    train_data_target[i]['label'] = agg_pred
        
        elif agg == 'BEA':
            # train_data_target = test_data
            idx2label = {v:k for k,v in self.args['label2idx'].items()}
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(train_data_target, best_model, self.criterion, 'cn2_extraction', task_id=task_id)[0]
                curr_pred = [curr_pred[i][:len(train_data_target[i]['label'])] for i in range(len(train_data_target))]
                preds.append([idx2label[r] for rr in curr_pred for r in rr])
            
            labels = [idx2label[r] for rr in train_data_target for r in rr['label']]
            
            all_labels = [r[0] for r in sorted(list(self.args['label2idx'].items()), key=lambda x:x[1])]
            if self.args['evaluator'] == 'token':
                all_tags = all_labels
                preds, _ = bea_ent_uns(labels, preds, all_labels, all_tags, False)
            else:
                all_tags = list(set([r.split('-')[-1] for r in all_labels]))
                preds, _ = bea_ent_uns(labels, preds, all_labels, all_tags, True)
            
            k = 0
            for i in range(len(train_data_target)):
                for j in range(len(train_data_target[i]['label'])):
                    train_data_target[i]['label'][j] = self.args['label2idx'][preds[k]]
                    k += 1
        
        else:
            assert False
        
        # Train 2nd phase with aggregated label
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        model_.aggregation_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data_target, model_, optimizer, 'cn2_aggregation')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            self.eval_task(test_data, model_, 'cn2_aggregation')
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_aggregation')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            score = self.eval_final(test_data, best_model, 'cn2_aggregation')
        else:
            score = self.eval_task(test_data, best_model, 'cn2_aggregation')
            
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    
    # train consensus network (alternated training extraction phase and aggregation phase)
    def train_cn2_alternate(self, train_data, dev_data, test_data, model_, best_model, exp):
        if self.args['cuda']:
           model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = None
        best_score = -np.Inf
        start_time = time.time()
        
        model_.extraction_phase()
        extraction_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        model_.aggregation_phase()
        aggregation_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        for epoch in range(1, self.args['epochs']+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs']))
            model_.extraction_phase()
            self.train_epoch(train_data, model_, extraction_optimizer, 'cn2_extraction')
            model_.aggregation_phase()
            self.train_epoch(train_data, model_, aggregation_optimizer, 'cn2_aggregation')
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_aggregation')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch = epoch
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_final(test_data, best_model, 'cn2_aggregation')
                    
        if exp == 1:
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    
    # train multi-task model
    def train_mtm(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None, agg='MV'):
        assert mode == 'leave-1-out'
        train_data = [r for r in train_data if r['task']!=target_task]
        dev_data = [r for r in dev_data if r['task']!=target_task] if dev_data else None
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn2_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn2_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        # AMV Label Aggregation
        if agg == 'MV':
            dev_scores = self.eval_each_task(dev_data, best_model, 'cn2_extraction')
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(test_data, best_model, self.criterion, 'cn2_extraction', task_id=task_id)[0]
                preds.append([dev_scores[task_id][1], curr_pred])
            
            final_pred = []
            for i in range(len(test_data)):
                pred_mat = np.zeros((self.args['tagset_size'], len(test_data[i]['label'])))
                for dev_score, curr_pred in preds:
                    curr_pred_i = curr_pred[i]
                    curr_pred_i = curr_pred_i[:len(test_data[i]['label'])]
                    pred_mat[curr_pred_i, range(len(curr_pred_i))] += dev_score
                
                final_pred.append(list(np.argmax(pred_mat, 0)))
        
        elif agg == 'BEA':
            # train_data_target = test_data
            idx2label = {v:k for k,v in self.args['label2idx'].items()}
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(test_data, best_model, self.criterion, 'cn2_extraction', task_id=task_id)[0]
                curr_pred = [curr_pred[i][:len(test_data[i]['label'])] for i in range(len(test_data))]
                preds.append([idx2label[r] for rr in curr_pred for r in rr])
            
            labels = [idx2label[r] for rr in test_data for r in rr['label']]
            
            all_labels = [r[0] for r in sorted(list(self.args['label2idx'].items()), key=lambda x:x[1])]
            if self.args['evaluator'] == 'token':
                all_tags = all_labels
                agg_pred, _ = bea_ent_uns(labels, preds, all_labels, all_tags, False)
            else:
                all_tags = list(set([r.split('-')[-1] for r in all_labels]))
                agg_pred, _ = bea_ent_uns(labels, preds, all_labels, all_tags, True)
            
            final_pred, curr_i = [], 0
            for i in range(len(test_data)):
                curr_pred = []
                for r in test_data[i]['label']:
                    curr_pred.append(self.args['label2idx'][agg_pred[curr_i]])
                    curr_i += 1
                
                final_pred.append(curr_pred)
            
        else:
            assert False
        
        acc, prec, recall, f1, loss = self.evaluator.evaluate(test_data, best_model, self.criterion, 'cn2_extraction', preds=final_pred)
        score = acc if self.metric=='acc' else f1
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    # train consensus network (two phase) in supervised and leave-1-out settings
    def train_mtm_mlp(self, train_data, dev_data, test_data, model_, best_model, exp, mode='supervised', target_task=None, agg='MV'):
        assert mode == 'leave-1-out'
        if self.args['cuda']:
            model_.cuda()
            best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'cn_mlp_extraction')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'cn_mlp_extraction')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        if mode == 'supervised':
            self.eval_final(test_data, best_model, 'cn_mlp_extraction')
        else:
            # self.eval_task(test_data, best_model, 'cn_mlp_extraction')
            print("No evaluation method available")
        
        if agg == 'MV':
            dev_scores = self.eval_each_task(dev_data, best_model, 'cn_mlp_extraction')
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(test_data, best_model, self.criterion, 'cn_mlp_extraction', task_id=task_id)[0]
                preds.append([dev_scores[task_id][1], curr_pred])
            
            preds_sum = np.array([r[1] for r in preds]).sum(0)
            final_preds = [1 if r else 0 for r in list(preds_sum > (len(preds)/2))]
        
        elif agg == 'BEA':
            preds = []
            for task_id in self.task2idx.values():
                if task_id == target_task:
                    continue
                curr_pred = self.evaluator.predictor.predict(test_data, best_model, self.criterion, 'cn_mlp_extraction', task_id=task_id)[0]
                preds.append(['T' if r==1 else 'F' for r in curr_pred])
            
            labels = ['T' if r['label']==1 else 'F' for r in test_data]
            all_labels = ['F', 'T']
            all_tags = ['F', 'T']
            final_preds, _ = bea_ent_uns(labels, preds, all_labels, all_tags, False)
            final_preds = [1 if r=='T' else 0 for r in final_preds]
        
        else:
            assert False
                
        
        acc, prec, recall, f1, loss = self.evaluator.evaluate(test_data, best_model, self.criterion, 'cn_mlp_aggregation', preds=final_preds)
        print("Performance on target task:")
        print("A: %.4f  P: %.4f  R: %.4f  F: %.4f\n" % (acc, prec, recall, f1))
        
        score = acc if self.metric=='acc' else f1
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    # train adversarial consensus network
    def train_cn2_adv(self, train_data, dev_data, test_data, model_, best_model, exp):
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # Extraction Phase
        model_.extraction_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        model_.train_cla_phase()
        train_cla_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        model_.adv_phase()
        adv_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['adv_lr'], momentum=0.9)
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            model_.extraction_phase()
            self.train_epoch(train_data, model_, optimizer, 'adv_cn-ext')
            model_.train_cla_phase()
            self.train_epoch(train_data, model_, train_cla_optimizer, 'adv_cn-train_cla')
            model_.adv_phase()
            self.train_epoch(train_data, model_, adv_optimizer, 'adv_cn-adv')
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'adv_cn-ext')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[0] = epoch
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        self.eval_final(test_data, best_model, 'adv_cn-ext')
        
        # Aggregation Phase
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        model_.aggregation_phase()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data, model_, optimizer, 'adv_cn-agg')
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'adv_cn-agg')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict())
                    best_epoch[1] = epoch
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_final(test_data, best_model, 'adv_cn-agg')
            
        if exp == 1:
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(self.args['epochs'][0])+"_"+str(self.args['epochs'][1])+"_"+"%.4f"%score+'.model')
    
    
    # train Peng et al, 2016 model in low-resource setting
    def train_Peng_2016_low(self, train_data, dev_data, test_data, model_, best_model, exp, target_task=None):
        test_data = [r for r in test_data if r['task']==target_task]
        if self.args['cuda']:
            model_, best_model = model_.cuda(), best_model.cuda()
        best_epoch = [None, None]
        best_score = -np.Inf
        start_time = time.time()
        
        # first step: train on all tasks
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][0]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][0]))
            self.train_epoch(train_data, model_, optimizer, 'Peng2016')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1
            
            if exp == 1:
                curr_score = self.eval_epoch(dev_data, model_, 'Peng2016')
                self.eval_task(test_data, model_, 'Peng2016')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch[0] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        self.eval_task(test_data, best_model, 'Peng2016')
        
        # second step: train on target task
        train_data = [r for r in train_data if r['task']==target_task]
        dev_data = [r for r in dev_data if r['task']==target_task] if dev_data else None
        best_score = -np.Inf
        model_.load_state_dict(best_model.state_dict())
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()), lr=self.args['lr'], momentum=0.9)
        patience = 0
        for epoch in range(1, self.args['epochs'][1]+1):
            print("Epoch: [%d/%d]\n" % (epoch, self.args['epochs'][1]))
            self.train_epoch(train_data, model_, optimizer, 'Peng2016')
            adjust_learning_rate(optimizer, self.args['lr'] / (1 + (epoch + 1) * self.args['lr_decay']))
            patience += 1

            if exp == 1:
                curr_score = self.eval_task(dev_data, model_, 'Peng2016')
                self.eval_task(test_data, model_, 'Peng2016')
                if curr_score >= best_score:
                    best_score = curr_score
                    best_model.load_state_dict(model_.state_dict()) # tested, no need to deepcopy!
                    best_epoch[1] = epoch
                    patience = 0
            
            if exp == 1 and patience > self.args['patience']:
                break
                            
            minutes = (time.time() - start_time) / 60
            print("Total time: %.4f minutes\n\n" % minutes)
        
        if exp == 2:
            best_model.load_state_dict(model_.state_dict())
        
        score = self.eval_task(test_data, best_model, 'Peng2016')
        
        if exp == 1:
            self.save_model(best_model, best_epoch, score, self.args['save_model']+"/Epoch"+str(best_epoch)+"_"+"%.4f"%score+'.model')
            return best_epoch
        if exp == 2:
            self.save_model(best_model, epoch, score, self.args['save_model']+"/Epoch"+str(epoch)+"_"+"%.4f"%score+'.model')
    
    
    def save_model(self, model_, epoch, score, path):
        torch.save({'epoch': epoch,
                    'args': self.args,
                    'state_dict': model_.state_dict(),
                    'score': score
                    # 'optimizer' : optimizer.state_dict()
                    }, path)