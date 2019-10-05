"""
.. module:: crf
    :synopsis: conditional random field
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.nn as nn
import model.utils as utils
import numpy as np

class CRF_L(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Ma et al. 2016, has more parameters than CRF_S
 
    args: 
        hidden_dim : input dim size 
        tagset_size: target_set_size 
        if_biase: whether allow bias in linear trans    
    """
    

    def __init__(self, hidden_dim, tagset_size, if_bias=True, sigmoid=""):
        assert sigmoid
        super(CRF_L, self).__init__()
        self.sigmoid = sigmoid
        self.tagset_size = tagset_size
        self.transitions = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size, bias=if_bias)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        utils.init_linear(self.transitions)

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        ins_num = feats.size(0) * feats.size(1)
        scores = self.hidden2tag(feats).view(ins_num, self.tagset_size, 1).expand(ins_num, self.tagset_size, self.tagset_size)
        trans_ = self.transitions(feats).view(ins_num, self.tagset_size, self.tagset_size)
        
        if self.sigmoid == "nosig":
            return scores + trans_
        elif self.sigmoid == "relu":
            return self.ReLU(scores + trans_)


class CRF_S_Base(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args: 
        rnn_hid: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
 
    """
    def __init__(self, rnn_hid, tagset_size):
        super(CRF_S_Base, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(rnn_hid*2, tagset_size)
        # transitions: each *i*, *j* means transition from *i* to *j*
        self.transitions = nn.Parameter(torch.Tensor(tagset_size, tagset_size))
        
        # utils.init_linear(self.hidden2tag)
        # self.transitions.data.zero_()
        nn.init.xavier_normal_(self.hidden2tag.weight)
        nn.init.xavier_normal_(self.transitions)


class Orig_CRF_S(CRF_S_Base):
    def __init__(self, rnn_hid, tagset_size):
        super(Orig_CRF_S, self).__init__(rnn_hid, tagset_size)
    
    def forward(self, crf_input):
        batch_size = crf_input.shape[0]
        max_seq_len = crf_input.shape[1]
        trans = self.transitions.unsqueeze(0).unsqueeze(0).repeat(batch_size,max_seq_len,1,1)
        
        scores = self.hidden2tag(crf_input)
        scores = scores.unsqueeze(2).repeat(1,1,self.tagset_size,1)
        
        crf_output = trans + scores
        
        return crf_output


class List_Orig_CRF_S(nn.Module):
    def __init__(self, rnn_hid, tagset_size, num_crfs):
        super(List_Orig_CRF_S, self).__init__()
        self.tagset_size = tagset_size
        hidden2tag_weights = torch.Tensor(1, rnn_hid*2, tagset_size)
        hidden2tag_bias = torch.Tensor(1, tagset_size)
        nn.init.xavier_normal_(hidden2tag_weights)
        nn.init.xavier_normal_(hidden2tag_bias)
        self.hidden2tag_weights = nn.Parameter(hidden2tag_weights.repeat(num_crfs,1,1))
        self.hidden2tag_bias = nn.Parameter(hidden2tag_bias.repeat(num_crfs,1))
        # transitions: each *i*, *j* means transition from *i* to *j*
        transitions = torch.Tensor(1, tagset_size, tagset_size)
        nn.init.xavier_normal_(transitions)
        self.transitions = nn.Parameter(transitions.repeat(num_crfs,1,1))
    
    def forward(self, crf_input, crf_ids):
        batch_size = crf_input.shape[0]
        max_seq_len = crf_input.shape[1]
        trans = self.transitions[crf_ids].unsqueeze(1).repeat(1,max_seq_len,1,1)
        h2t_weights = self.hidden2tag_weights[crf_ids]
        h2t_bias = self.hidden2tag_bias[crf_ids].unsqueeze(1).repeat(1,max_seq_len,1)
        scores = crf_input.bmm(h2t_weights) + h2t_bias
        scores = scores.unsqueeze(2).repeat(1,1,self.tagset_size,1)
        
        crf_output = trans + scores
        
        return crf_output


class Crowd_add_CRF_S(CRF_S_Base):
    def __init__(self, rnn_hid, tagset_size):
        super(Crowd_add_CRF_S, self).__init__(rnn_hid, tagset_size)

    def forward(self, crf_input, crowd_rep):
        batch_size = crf_input.shape[0]
        max_seq_len = crf_input.shape[1]
        trans = self.transitions.unsqueeze(0).unsqueeze(0).repeat(batch_size,max_seq_len,1,1)
        
        scores = self.hidden2tag(crf_input)
        scores = scores + crowd_rep.unsqueeze(1).repeat(1, max_seq_len, 1)
        scores = scores.unsqueeze(2).repeat(1,1,self.tagset_size,1)
        
        crf_output = trans + scores
        
        return crf_output


class CN_CRF_S(CRF_S_Base):
    def __init__(self, rnn_hid, tagset_size):
        super(CN_CRF_S, self).__init__(rnn_hid, tagset_size)
    
    def extraction_phase(self):
        self.hidden2tag.weight.requires_grad = True
        self.transitions.requires_grad = True
    
    def aggregation_phase(self):
        self.hidden2tag.weight.requires_grad = False
        self.transitions.requires_grad = False
    
    def forward(self, crf_input, consensus):
        batch_size = crf_input.shape[0]
        max_seq_len = crf_input.shape[1]
        trans = self.transitions.unsqueeze(0).repeat(batch_size,1,1).bmm(consensus)
        trans = trans.unsqueeze(1).repeat(1,max_seq_len,1,1)
        
        scores = self.hidden2tag(crf_input)
        scores = scores.bmm(consensus)
        scores = scores.unsqueeze(2).repeat(1,1,self.tagset_size,1)
        
        crf_output = trans + scores
        
        return crf_output


class FCN_CRF_S(CRF_S_Base):
    def __init__(self, rnn_hid, tagset_size):
        super(FCN_CRF_S, self).__init__(rnn_hid, tagset_size)
    
    def extraction_phase(self):
        self.hidden2tag.weight.requires_grad = True
        self.transitions.requires_grad = True
    
    def aggregation_phase(self):
        self.hidden2tag.weight.requires_grad = False
        self.transitions.requires_grad = False
    
    def forward(self, crf_input, consensus1, consensus2):
        batch_size = crf_input.shape[0]
        max_seq_len = crf_input.shape[1]
        trans = self.transitions.unsqueeze(0).repeat(batch_size,1,1).bmm(consensus1).bmm(consensus2)
        trans = trans.unsqueeze(1).repeat(1,max_seq_len,1,1)
        
        scores = self.hidden2tag(crf_input)
        scores = scores.bmm(consensus1).bmm(consensus2)
        scores = scores.unsqueeze(2).repeat(1,1,self.tagset_size,1)
        
        crf_output = trans + scores
        
        return crf_output


class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch
    
    def calc_energy_gold_ts(self, scores, target, mask):
        # calculate energy (unnormalized log proba) of the gold tag sequence
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target.unsqueeze(-1)).squeeze(-1)  # seq_len * bat_size
        # tg_energy = tg_energy.masked_select(mask).sum()
        tg_energy = (tg_energy * mask.float()).sum(0)
        
        return tg_energy
    
    def forward_algo(self, scores, mask):
        # Forward Algorithm
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        cur_partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        partition = cur_partition
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target            
            cur_values = cur_values + cur_partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition.contiguous(), cur_partition.contiguous(),
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
        
        #only need end at end_tag
        # partition = partition[:, self.end_tag].sum()
        partition = partition[:, self.end_tag]
        
        return partition
    
    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
            *idea ("Li", "P11", "P12", "P21"...): idea for training (loss calculation)
        return:
            loss
        """
        bat_size = scores.size(1)
        
        # numerator and denominator: ...of the likelihood function:)
        numerator = self.calc_energy_gold_ts(scores, target, mask)
        denominator = self.forward_algo(scores, mask)
        loss = denominator - numerator
        
        # average_batch
        if self.average_batch:
            loss = loss / bat_size
        
        return loss


class CRFDecode_vb():
    """Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag


    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask
        #decode_idx = scores.new(seq_len-1, bat_size).long()
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer.squeeze(-1)
        return decode_idx