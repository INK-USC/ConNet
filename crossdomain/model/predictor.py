import torch
import numpy as np
from model.data_utils import *


class Predictor_CRF:
    def __init__(self, args, decoder, task2idx):
        self.args = args
        self.decoder = decoder
        self.task2idx = task2idx
        
    def predict(self, data, model_, criterion, model_name, task_id=None):
        preds, losses = [], []
        model_.eval()
        if model_name == 'Peng2016':
            data = pack_data_tasks(self.args, data, self.task2idx)
        else:
            data = pack_data(self.args, data)
        for batch in data:
            sent_end, task_ids = batch['sent_end'], batch['task_ids']
            if task_id:
                task_ids = [task_id for i in range(len(batch['task_ids']))]
            model_.zero_grad()
            if model_name == 'stm':
                crf_out = model_(batch['data_input'])
            elif model_name == 'mtm':
                crf_out = model_(batch['data_input'], task_ids)
            elif model_name == 'crowd_add':
                crf_out = model_(batch['data_input'], task_ids)
            elif model_name == 'crowd_add_test':
                crf_out = model_(batch['data_input'], None)
            elif model_name == 'cn2_extraction':
                crf_out = model_(batch['data_input'], None, task_ids)
            elif model_name == 'cn2_aggregation':
                crf_out = model_(batch['data_input'], sent_end, None)
            elif model_name == 'adv_cn-ext':
                crf_out = model_(batch['data_input'], None, task_ids, model_name)
            elif model_name == 'adv_cn-agg':
                crf_out = model_(batch['data_input'], sent_end, None, model_name)
            elif model_name == 'Peng2016':
                crf_out = model_(batch['data_input'], task_ids[0])
            else:
                assert False
            
            crf_out = crf_out.transpose(0,1)
            mask = batch['mask'].transpose(0,1)
            label = batch['label'].transpose(0,1)
            loss = criterion(crf_out, label, mask)
            loss = loss.sum().cpu().tolist()
            decoded = self.decoder.decode(crf_out, mask).transpose(0,1)
            preds += decoded.cpu().tolist()
            losses.append(loss)
        
        return preds, np.mean(losses)


class Predictor_Binary:
    def __init__(self, args):
        self.args = args
        
    def predict(self, data, model_, criterion, model_name, task_id=None):
        preds, losses = [], []
        model_.eval()
        data = pack_data_mlp(self.args, data)
        for batch in data:
            task_ids = batch['task_ids']
            if task_id:
                task_ids = [task_id for i in range(len(batch['task_ids']))]
            model_.zero_grad()
            if model_name == 'stm_mlp':
                crf_out = model_(batch['data_input'])
            elif model_name == 'cn_mlp_extraction':
                crf_out = model_(batch['data_input'], task_ids)
            elif model_name == 'cn_mlp_aggregation':
                crf_out = model_(batch['data_input'], None)
            elif model_name == 'crowd_add_mlp':
                crf_out = model_(batch['data_input'], task_ids)
            elif model_name == 'crowd_add_mlp_test':
                crf_out = model_(batch['data_input'], None)
            else:
                assert False
            
            loss = criterion(crf_out, batch['label'])
            loss = loss.cpu().tolist()
            preds += (crf_out >= 0.5).cpu().tolist()
            losses.append(loss)
        
        return preds, np.mean(losses)