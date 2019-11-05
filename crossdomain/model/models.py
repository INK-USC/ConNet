import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import model.crf as crf


# Single-task model with MLP
class STM_MLP(nn.Module):
    def __init__(self, args):
        
        super(STM_MLP, self).__init__()
        
        self.mlp = nn.Linear(5000, args['mlp_hid'])
        self.cla = nn.Linear(args['mlp_hid'], 1)
        self.dropout = nn.Dropout(p=args['dropout'])
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data_input):
        mlp_out = self.dropout(self.mlp(data_input))
        cla = self.sigmoid(self.cla(mlp_out)).squeeze(1)
        
        return cla

# CN_MLP model
class CN_MLP(nn.Module):
    def __init__(self, args):
        
        super(CN_MLP, self).__init__()
        
        self.args = args
        
        self.mlp = nn.Linear(5000, args['mlp_hid'])
        
        self.CM = nn.Parameter(torch.Tensor(args['num_tasks'], args['mlp_hid'], args['mlp_hid']))
        nn.init.xavier_normal_(self.CM)
        
        self.attn = nn.Parameter(torch.Tensor(args['mlp_hid'], args['num_tasks']))
        nn.init.xavier_normal_(self.attn)
        
        self.cla = nn.Linear(args['mlp_hid'], 1)
        
        self.dropout = nn.Dropout(p=args['dropout'])
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    # switch to extraction training
    def extraction_phase(self):
        self.attn.requires_grad = False
        self.mlp.weight.requires_grad = True
        self.CM.requires_grad = True
        self.cla.weight.requires_grad = True
    
    # switch to aggregation training
    def aggregation_phase(self):
        self.attn.requires_grad = True
        if not self.args['fine_tune']:
            self.mlp.weight.requires_grad = False
            self.CM.requires_grad = False
            self.cla.weight.requires_grad = False
    
    def forward(self, data_input, task_ids):
        # process the input and return lstm output
        # data_input: batch_size * 5000
        
        mlp_out = self.dropout(self.mlp(data_input))
        
        # aggregation phase
        if task_ids == None:
            task_attn = mlp_out.mm(self.attn)
            batch_size = mlp_out.shape[0]
            consensus = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM, 0).repeat(batch_size,1,1,1)).sum(1)
        # extraction phase
        else:
            consensus = self.CM[task_ids]
        
        mlp_out_1 = self.dropout(mlp_out.unsqueeze(1).bmm(consensus)).squeeze(1)
        cla = self.sigmoid(self.cla(mlp_out_1)).squeeze(1)
        
        return cla


# base class for all models: take input data, return lstm output
class Base_Model(nn.Module):
    def __init__(self, args, emb):
        
        super(Base_Model, self).__init__()
        
        self.args = args
        
        # embeddings, lstms
        if self.args['char_rnn']:
            self.char_emb = nn.Embedding.from_pretrained(Parameter(torch.FloatTensor(emb[0])), freeze=False if args['fine_tune_char_emb'] else True)
            self.char_lstm = nn.LSTM(self.char_emb.weight.shape[1], args['char_rnn_hid'], args['char_rnn_layers'], bidirectional=True, batch_first=True)
        if self.args['word_rnn']:
            self.word_emb = nn.Embedding.from_pretrained(Parameter(torch.FloatTensor(emb[1])), freeze=False if args['fine_tune_word_emb'] else True)
            word_rnn_input_dim = self.word_emb.weight.shape[1]
            if self.args['char_rnn']:
                word_rnn_input_dim += 2 * args['char_rnn_hid']
            self.word_lstm = nn.LSTM(word_rnn_input_dim, args['word_rnn_hid'], args['word_rnn_layers'], bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(p=args['dropout'])
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, data_input):
        # process the input and return lstm output
        # chars (data_input[0]): batch_size * char_seq_len * char_emb_dim
        # words (data_input[1]): batch_size * word_seq_len * word_emb_dim
        
        if self.args['char_rnn']:
            chars, char_start, char_end = data_input[0]
            batch_size = chars.shape[0]
            char_emb_out = self.dropout(self.char_emb(chars))
            char_lstm_out = self.dropout(self.char_lstm(char_emb_out)[0])
            if self.args['word_rnn']:
                char_lstm_out_f = char_lstm_out[:,:,:self.args['char_rnn_hid']]
                char_lstm_out_b = char_lstm_out[:,:,self.args['char_rnn_hid']:]
                word_max_len = np.array(char_start).shape[1]
                idx = [[ii for i in range(word_max_len)] for ii in range(batch_size)]
                word_rep_f = char_lstm_out_f[[idx,char_end]]
                word_rep_b = char_lstm_out_b[[idx,char_start]]
                word_rep = torch.cat((word_rep_f, word_rep_b), dim=2)
        if self.args['word_rnn']:
            words = data_input[1]
            batch_size = words.shape[0]
            word_emb_out = self.dropout(self.word_emb(words))
            if self.args['char_rnn']:
                mask = words != self.args['pad_word_idx']
                word_rep = word_rep * mask.float().unsqueeze(-1)
                word_out = torch.cat((word_emb_out, word_rep), dim=2)
            else:
                word_out = word_emb_out
            lstm_out = self.dropout(self.word_lstm(word_out)[0])
        
        return lstm_out


# Single-task model
class STM(Base_Model):
    def __init__(self, args, emb):
        
        super(STM, self).__init__(args, emb)
        
        self.crf = crf.Orig_CRF_S(args['word_rnn_hid'], args['tagset_size'])
    
    def forward(self, data_input):
        lstm_out = super(STM, self).forward(data_input)
        
        crf_out = self.crf(lstm_out)
        
        return crf_out


# Multi-task model
class MTM(Base_Model):
    def __init__(self, args, emb):
        
        super(MTM, self).__init__(args, emb)
        
        self.crf_list = crf.List_Orig_CRF_S(args['word_rnn_hid'], args['tagset_size'], args['num_tasks'])
    
    def forward(self, data_input, task_ids):
        lstm_out = super(MTM, self).forward(data_input)
        
        crf_out = self.crf_list(lstm_out, task_ids)
        
        return crf_out


# Peng et al, 2016
class Peng_2016(Base_Model):
    def __init__(self, args, emb, method):
        assert method in ['domain_mask', 'domain_trans']
        
        super(Peng_2016, self).__init__(args, emb)
        
        self.method = method
        if self.method == 'domain_mask':
            self.domain_width = int(self.args['word_rnn_hid'] / (self.args['num_tasks']+1))
            self.shared_width = args['word_rnn_hid'] - self.domain_width * self.args['num_tasks']
        else:
            self.trans_mat = nn.Parameter(torch.Tensor(self.args['num_tasks'], args['word_rnn_hid']*2, args['word_rnn_hid']*2))
            nn.init.xavier_normal_(self.trans_mat)
        
        self.crf = crf.Orig_CRF_S(args['word_rnn_hid'], args['tagset_size'])
    
    def forward(self, data_input, task_id):
        lstm_out = super(Peng_2016, self).forward(data_input)
        
        if self.method == 'domain_mask':
            domain_mask = np.zeros(self.args['word_rnn_hid']*2)
            domain_mask[:self.shared_width] = 1
            domain_mask[self.args['word_rnn_hid']:(self.args['word_rnn_hid']+self.shared_width)] = 1
            domain_mask[(self.shared_width+self.domain_width*task_id):(self.shared_width+self.domain_width*(task_id+1))] = 1
            domain_mask[(self.args['word_rnn_hid']+self.shared_width+self.domain_width*task_id):(self.args['word_rnn_hid']+self.shared_width+self.domain_width*(task_id+1))] = 1
            domain_mask = torch.tensor(domain_mask)
            if self.args['cuda']:
                domain_mask = domain_mask.cuda()
            crf_input = lstm_out * domain_mask.float().unsqueeze(0).unsqueeze(0)
            crf_out = self.crf(crf_input)
        else:
            batch_size = lstm_out.shape[0]
            batch_trans_mat = self.trans_mat[task_id].unsqueeze(0).repeat(batch_size, 1, 1)
            crf_input = torch.bmm(lstm_out, batch_trans_mat)
            crf_out = self.crf(crf_input)
        
        return crf_out


# Crowd-Add
class Crowd_add(Base_Model):
    def __init__(self, args, emb):
        
        super(Crowd_add, self).__init__(args, emb)
        
        self.crowd_vecs = nn.Parameter(torch.Tensor(args['num_tasks'], args['tagset_size']))
        nn.init.xavier_normal_(self.crowd_vecs)
        self.crf = crf.Crowd_add_CRF_S(args['word_rnn_hid'], args['tagset_size'])
    
    
    def forward(self, data_input, task_ids):
        lstm_out = super(Crowd_add, self).forward(data_input)
        batch_size = lstm_out.shape[0]
        if task_ids:
            crowd_rep = self.crowd_vecs[task_ids]
        else:
            crowd_rep = self.crowd_vecs.mean(0).unsqueeze(0).repeat(batch_size, 1)
        
        crf_out = self.crf(lstm_out, crowd_rep)
        
        return crf_out


# Crowd-Add
class Crowd_add_MLP(nn.Module):
    def __init__(self, args):
        
        super(Crowd_add_MLP, self).__init__()
        
        self.mlp = nn.Linear(5000, args['mlp_hid'])
        self.crowd_vecs = nn.Parameter(torch.Tensor(args['num_tasks'], args['mlp_hid']))
        nn.init.xavier_normal_(self.crowd_vecs)
        self.cla = nn.Linear(args['mlp_hid'], 1)
        self.dropout = nn.Dropout(p=args['dropout'])
        self.sigmoid = nn.Sigmoid()    
    
    def forward(self, data_input, task_ids):
        mlp_out = self.dropout(self.mlp(data_input))
        batch_size = mlp_out.shape[0]
        if task_ids:
            crowd_rep = self.crowd_vecs[task_ids]
        else:
            crowd_rep = self.crowd_vecs.mean(0).unsqueeze(0).repeat(batch_size, 1)
        
        mlp_out_ = mlp_out + crowd_rep
        cla = self.sigmoid(self.cla(mlp_out)).squeeze(1)
        
        return cla


# Crowd-Add
class Crowd_cat(Base_Model):
    def __init__(self, args, emb):
        
        super(Crowd_cat, self).__init__(args, emb)
        
        self.crowd_vecs = nn.Parameter(torch.Tensor(args['num_tasks'], 50))
        nn.init.xavier_normal_(self.crowd_vecs)
        self.crf = crf.Orig_CRF_S(args['word_rnn_hid']+25, args['tagset_size'])
    
    
    def forward(self, data_input, task_ids):
        lstm_out = super(Crowd_cat, self).forward(data_input)
        batch_size = lstm_out.shape[0]
        if task_ids:
            crowd_rep = self.crowd_vecs[task_ids]
        else:
            crowd_rep = self.crowd_vecs.mean(0).unsqueeze(0).repeat(batch_size, 1)
        lstm_out_ = torch.cat((lstm_out, crowd_rep.unsqueeze(1).repeat(1,lstm_out.shape[1],1)), 2)
        crf_out = self.crf(lstm_out_)
        
        return crf_out


# Crowd-Add
class Crowd_cat_MLP(nn.Module):
    def __init__(self, args):
        
        super(Crowd_cat_MLP, self).__init__()
        
        self.mlp = nn.Linear(5000, args['mlp_hid'])
        self.crowd_vecs = nn.Parameter(torch.Tensor(args['num_tasks'], 50))
        nn.init.xavier_normal_(self.crowd_vecs)
        self.cla = nn.Linear(args['mlp_hid']+50, 1)
        self.dropout = nn.Dropout(p=args['dropout'])
        self.sigmoid = nn.Sigmoid()    
    
    def forward(self, data_input, task_ids):
        mlp_out = self.dropout(self.mlp(data_input))
        batch_size = mlp_out.shape[0]
        if task_ids:
            crowd_rep = self.crowd_vecs[task_ids]
        else:
            crowd_rep = self.crowd_vecs.mean(0).unsqueeze(0).repeat(batch_size, 1)
        
        mlp_out_ = torch.cat((mlp_out, crowd_rep), 1)
        cla = self.sigmoid(self.cla(mlp_out_)).squeeze(1)
        
        return cla


# base class for two-phase training Consensus Network
class Base_CN(Base_Model):
    def __init__(self, args, emb):
        
        super(Base_CN, self).__init__(args, emb)
        
        # attention layer
        self.attn = nn.Parameter(torch.Tensor(args['word_rnn_hid']*2, args['num_tasks']))
        nn.init.xavier_normal_(self.attn)
    
    # switch to extraction training
    def extraction_phase(self):
        self.attn.requires_grad = False
        if self.args['char_rnn']:
            if self.args['fine_tune_char_emb']:
                self.char_emb.weight.requires_grad = True
            for param in self.char_lstm.parameters():
                param.requires_grad = True
        if self.args['word_rnn']:
            if self.args['fine_tune_word_emb']:
                self.word_emb.weight.requires_grad = True
            for param in self.word_lstm.parameters():
                param.requires_grad = True
    
    # switch to aggregation training
    def aggregation_phase(self):
        self.attn.requires_grad = True
        if not self.args['fine_tune']:
            if self.args['char_rnn']:
                self.char_emb.weight.requires_grad = False
                for param in self.char_lstm.parameters():
                    param.requires_grad = False
            if self.args['word_rnn']:
                self.word_emb.weight.requires_grad = False
                for param in self.word_lstm.parameters():
                    param.requires_grad = False
    
    def forward(self, data_input, sent_end):
        # sent_end: batch_size or None (in extraction phase)
        
        lstm_out = super(Base_CN, self).forward(data_input)
        batch_size = lstm_out.shape[0]
        
        # compute attention output for aggregation phase
        if sent_end:
            lstm_out_f = lstm_out[:,:,:int(lstm_out.shape[2]/2)]
            lstm_out_b = lstm_out[:,:,int(lstm_out.shape[2]/2):]
            f_sent_rep = lstm_out_f[[range(batch_size),sent_end]]
            b_sent_rep = lstm_out_b[[range(batch_size),[0]*batch_size]]
            sent_rep = torch.cat((f_sent_rep, b_sent_rep), dim=1)
            task_attn = sent_rep.mm(self.attn)
            return lstm_out, task_attn
        else:
            return lstm_out


# Two-phase training Consensus Network
class CN_2(Base_CN):
    def __init__(self, args, emb):
        
        super(CN_2, self).__init__(args, emb)
        
        self.crf = crf.CN_CRF_S(args['word_rnn_hid'], args['tagset_size'])
        
        # consensus matrices
        self.CM = nn.Parameter(torch.Tensor(args['num_tasks'], args['tagset_size'], args['tagset_size']))
        nn.init.xavier_normal_(self.CM)
    
    def extraction_phase(self):
        super(CN_2, self).extraction_phase()
        self.CM.requires_grad = True
        self.crf.extraction_phase()
    
    def aggregation_phase(self):
        super(CN_2, self).aggregation_phase()
        if not self.args['fine_tune']:
            self.CM.requires_grad = False
            self.crf.aggregation_phase()
    
    def forward(self, data_input, sent_end, task_ids):
        # aggregation phase
        if task_ids == None:
            lstm_out, task_attn = super(CN_2, self).forward(data_input, sent_end)
            batch_size = lstm_out.shape[0]
            consensus = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM, 0).repeat(batch_size,1,1,1)).sum(1)
        # extraction phase
        else:
            lstm_out = super(CN_2, self).forward(data_input, None)
            consensus = self.CM[task_ids]
        
        crf_out = self.crf(lstm_out, consensus)
        
        return crf_out


# Flexible Consensus Network (two linear mappings, both task-specific)
class FCN_2(Base_CN):
    def __init__(self, args, emb):
        
        super(FCN_2, self).__init__(args, emb)
        
        self.crf = crf.FCN_CRF_S(args['word_rnn_hid'], args['tagset_size'])
        
        # consensus matrices
        self.CM1 = nn.Parameter(torch.Tensor(args['num_tasks'], args['tagset_size'], args['cm_dim']))
        self.CM2 = nn.Parameter(torch.Tensor(args['num_tasks'], args['cm_dim'], args['tagset_size']))
        nn.init.xavier_normal_(self.CM1)
        nn.init.xavier_normal_(self.CM2)
    
    def extraction_phase(self):
        super(FCN_2, self).extraction_phase()
        self.CM1.requires_grad = True
        self.CM2.requires_grad = True
        self.crf.extraction_phase()
    
    def aggregation_phase(self):
        super(FCN_2, self).aggregation_phase()
        if not self.args['fine_tune']:
            self.CM1.requires_grad = False
            self.CM2.requires_grad = False
            self.crf.aggregation_phase()
    
    def forward(self, data_input, sent_end, task_ids):
        # aggregation phase
        if task_ids == None:
            lstm_out, task_attn = super(FCN_2, self).forward(data_input, sent_end)
            batch_size = lstm_out.shape[0]
            consensus1 = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM1, 0).repeat(batch_size,1,1,1)).sum(1)
            consensus2 = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM2, 0).repeat(batch_size,1,1,1)).sum(1)
        # extraction phase
        else:
            lstm_out = super(FCN_2, self).forward(data_input, None)
            consensus1 = self.CM1[task_ids]
            consensus2 = self.CM2[task_ids]
        
        crf_out = self.crf(lstm_out, consensus1, consensus2)
        
        return crf_out


# More Flexible Consensus Network v2 (first mapping shared, second mapping task-specific)
class FCN_2_v2(Base_CN):
    def __init__(self, args, emb):
        
        super(FCN_2_v2, self).__init__(args, emb)
        
        self.crf = crf.FCN_CRF_S(args['word_rnn_hid'], args['tagset_size'])
        
        # consensus matrices
        self.CM1 = nn.Parameter(torch.Tensor(args['tagset_size'], args['cm_dim']))
        self.CM2 = nn.Parameter(torch.Tensor(args['num_tasks'], args['cm_dim'], args['tagset_size']))
        nn.init.xavier_normal_(self.CM1)
        nn.init.xavier_normal_(self.CM2)
    
    def extraction_phase(self):
        super(FCN_2_v2, self).extraction_phase()
        self.CM1.requires_grad = True
        self.CM2.requires_grad = True
        self.crf.extraction_phase()
    
    def aggregation_phase(self):
        super(FCN_2_v2, self).aggregation_phase()
        if not self.args['fine_tune']:
            self.CM1.requires_grad = False
            self.CM2.requires_grad = False
            self.crf.aggregation_phase()
    
    def forward(self, data_input, sent_end, task_ids):
        # aggregation phase
        if task_ids == None:
            lstm_out, task_attn = super(FCN_2_v2, self).forward(data_input, sent_end)
            batch_size = lstm_out.shape[0]
            consensus2 = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM2, 0).repeat(batch_size,1,1,1)).sum(1)
        # extraction phase
        else:
            lstm_out = super(FCN_2_v2, self).forward(data_input, None)
            batch_size = lstm_out.shape[0]
            consensus2 = self.CM2[task_ids]
        
        # first consensus is shared
        consensus1 = self.CM1.unsqueeze(0).repeat(batch_size, 1, 1)
        crf_out = self.crf(lstm_out, consensus1, consensus2)
        
        return crf_out


# Base class for adversarial training Consensus Network
class Base_ADV_CN(Base_CN):
    def __init__(self, args, emb):
        super(Base_ADV_CN, self).__init__(args, emb)
        
        # classifier (for adversarial training)
        self.cla = nn.Linear(args['word_rnn_hid']*2, args['num_tasks'])
        nn.init.xavier_normal_(self.cla.weight)
    
    def extraction_phase(self):
        super(Base_ADV_CN, self).extraction_phase()
        self.cla.weight.requires_grad = False
    
    def aggregation_phase(self):
        super(Base_ADV_CN, self).aggregation_phase()
        self.cla.weight.requires_grad = False
    
    def train_cla_phase(self):
        self.cla.weight.requires_grad = True
        self.attn.requires_grad = False
        if self.args['char_rnn']:
            self.char_emb.weight.requires_grad = False
            for param in self.char_lstm.parameters():
                param.requires_grad = False
        if self.args['word_rnn']:
            self.word_emb.weight.requires_grad = False
            for param in self.word_lstm.parameters():
                param.requires_grad = False
    
    def adv_phase(self):
        self.cla.weight.requires_grad = False
        self.attn.requires_grad = False
        if self.args['char_rnn']:
            self.char_emb.weight.requires_grad = False
            for param in self.char_lstm.parameters():
                param.requires_grad = True
        if self.args['word_rnn']:
            self.word_emb.weight.requires_grad = False
            for param in self.word_lstm.parameters():
                param.requires_grad = True
    
    def forward(self, data_input, sent_end, model_name):
        # aggregation phase
        if model_name == 'adv_cn-agg':
            lstm_out, task_attn = super(Base_ADV_CN, self).forward(data_input, sent_end)
            return lstm_out, task_attn
        # extraction phase
        elif model_name == 'adv_cn-ext':
            lstm_out = super(Base_ADV_CN, self).forward(data_input, None)
            return lstm_out
        # training classifier phase or adversarial training phase
        elif model_name in ['adv_cn-adv', 'adv_cn-train_cla']:
            lstm_out = super(Base_ADV_CN, self).forward(data_input, None)
            batch_size = lstm_out.shape[0]
            lstm_out_f = lstm_out[:,:,:self.args['word_rnn_hid']]
            lstm_out_b = lstm_out[:,:,self.args['word_rnn_hid']:]
            f_sent_rep = lstm_out_f[[range(batch_size),sent_end]]
            b_sent_rep = lstm_out_b[[range(batch_size),[0]*batch_size]]
            sent_rep = torch.cat((f_sent_rep, b_sent_rep), dim=1)
            cla_out = self.cla(sent_rep)
            return cla_out
        

class ADV_CN(Base_ADV_CN):
    def __init__(self, args, emb):
        
        super(ADV_CN, self).__init__(args, emb)
        
        self.crf = crf.CN_CRF_S(args['word_rnn_hid'], args['tagset_size'])
        
        # consensus matrices
        self.CM = nn.Parameter(torch.Tensor(args['num_tasks'], args['tagset_size'], args['tagset_size']))
        nn.init.xavier_normal_(self.CM)
    
    def extraction_phase(self):
        super(ADV_CN, self).extraction_phase()
        self.CM.requires_grad = True
        self.crf.extraction_phase()
    
    def aggregation_phase(self):
        super(ADV_CN, self).aggregation_phase()
        if self.args['fine_tune']:
            self.CM.requires_grad = True
            self.crf.extraction_phase()
        else:
            self.CM.requires_grad = False
            self.crf.aggregation_phase()
    
    def train_cla_phase(self):
        super(ADV_CN, self).train_cla_phase()
        self.CM.requires_grad = False
        self.crf.aggregation_phase()
    
    def adv_phase(self):
        super(ADV_CN, self).adv_phase()
        self.CM.requires_grad = False
        self.crf.aggregation_phase()
    
    def forward(self, data_input, sent_end, task_ids, model_name):
        if model_name == 'adv_cn-agg':
            lstm_out, task_attn = super(ADV_CN, self).forward(data_input, sent_end, model_name)
            batch_size = lstm_out.shape[0]
            consensus = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM, 0).repeat(batch_size,1,1,1)).sum(1)
            crf_out = self.crf(lstm_out, consensus)
            return crf_out
            
        elif model_name == 'adv_cn-ext':
            lstm_out = super(ADV_CN, self).forward(data_input, None, model_name)
            consensus = self.CM[task_ids]
            crf_out = self.crf(lstm_out, consensus)
            return crf_out
        
        elif model_name in ['adv_cn-adv', 'adv_cn-train_cla']:
            cla_out = super(ADV_CN, self).forward(data_input, sent_end, model_name)
            return cla_out
        

class ADV_FCN_v2(Base_ADV_CN):
    def __init__(self, args, emb):
        
        super(ADV_FCN_v2, self).__init__(args, emb)
        
        self.crf = crf.FCN_CRF_S(args['word_rnn_hid'], args['tagset_size'])
        
        # consensus matrices
        self.CM1 = nn.Parameter(torch.Tensor(args['tagset_size'], args['cm_dim']))
        self.CM2 = nn.Parameter(torch.Tensor(args['num_tasks'], args['cm_dim'], args['tagset_size']))
        nn.init.xavier_normal_(self.CM1)
        nn.init.xavier_normal_(self.CM2)
    
    def extraction_phase(self):
        super(ADV_FCN_v2, self).extraction_phase()
        self.CM1.requires_grad = True
        self.CM2.requires_grad = True
        self.crf.extraction_phase()
    
    def aggregation_phase(self):
        super(ADV_FCN_v2, self).aggregation_phase()
        if self.args['fine_tune']:
            self.CM1.requires_grad = True
            self.CM2.requires_grad = True
            self.crf.extraction_phase()
        else:
            self.CM1.requires_grad = False
            self.CM2.requires_grad = False
            self.crf.aggregation_phase()
    
    def train_cla_phase(self):
        super(ADV_FCN_v2, self).train_cla_phase()
        self.CM1.requires_grad = False
        self.CM2.requires_grad = False
        self.crf.aggregation_phase()
    
    def adv_phase(self):
        super(ADV_FCN_v2, self).adv_phase()
        self.CM1.requires_grad = False
        self.CM2.requires_grad = False
        self.crf.aggregation_phase()
    
    def forward(self, data_input, sent_end, task_ids, model_name):        
        if model_name == 'adv_cn-agg':
            lstm_out, task_attn = super(ADV_FCN_v2, self).forward(data_input, sent_end, model_name)
            batch_size = lstm_out.shape[0]
            consensus1 = self.CM1.unsqueeze(0).repeat(batch_size, 1, 1)
            consensus2 = (self.softmax(task_attn).unsqueeze(-1).unsqueeze(-1) * torch.unsqueeze(self.CM2, 0).repeat(batch_size,1,1,1)).sum(1)
            crf_out = self.crf(lstm_out, consensus1, consensus2)
            return crf_out
        
        elif model_name == 'adv_cn-ext':
            lstm_out = super(ADV_FCN_v2, self).forward(data_input, None, model_name)
            batch_size = lstm_out.shape[0]
            consensus1 = self.CM1.unsqueeze(0).repeat(batch_size, 1, 1)
            consensus2 = self.CM2[task_ids]
            crf_out = self.crf(lstm_out, consensus1, consensus2)
            return crf_out
        
        elif model_name in ['adv_cn-adv', 'adv_cn-train_cla']:
            cla_out = super(ADV_FCN_v2, self).forward(data_input, sent_end, model_name)
            return cla_out


