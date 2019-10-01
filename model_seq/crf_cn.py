"""
.. module:: crf
    :synopsis: conditional random field
 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
import model_seq.utils as utils
import numpy as np
import sys
from collections import OrderedDict

class CRF(nn.Module):
    """
    Conditional Random Field Module

    Parameters
    ----------
    hidden_dim : ``int``, required.
        the dimension of the input features.
    tagset_size : ``int``, required.
        the size of the target labels.
    a_num : ``int``, required.
        the number of annotators.
    task : ``str``, required.
        the model task
    if_bias: ``bool``, optional, (default=True).
        whether the linear transformation has the bias term.
    """
    def __init__(self, 
                hidden_dim: int, 
                tagset_size: int, 
                a_num: int,
                task: str,
                test_att: bool = False,
                train_att: bool = False,
                if_bias: bool = True):
        super(CRF, self).__init__()

        self.tagset_size = tagset_size
        self.a_num = a_num
        self.task = task
        self.hidden_dim = hidden_dim
        self.test_att = test_att
        self.train_att = train_att

        # crowd components
        if 'maMulVecCrowd' in self.task:
            self.maMulVecCrowd = nn.Parameter(torch.Tensor(self.a_num, self.tagset_size))
        if 'maAddVecCrowd' in self.task:
            self.maAddVecCrowd = nn.Parameter(torch.Tensor(self.a_num, self.tagset_size))
        if 'maCatVecCrowd' in self.task:
            self.maCatVecCrowd = nn.Parameter(torch.Tensor(self.a_num, self.tagset_size))
            self.maCatVecCrowd_latent = nn.Parameter(torch.Tensor(self.tagset_size))
        if 'maMulMatCrowd' in self.task:
            self.maMulMatCrowd = nn.Parameter(torch.Tensor(self.a_num, self.tagset_size, self.tagset_size))
        if 'maMulCRFCrowd' in self.task:
            self.maMulCRFCrowd = nn.Parameter(torch.Tensor(self.a_num, self.tagset_size, self.tagset_size))
        if 'maMulScoreCrowd' in self.task:
            self.maMulScoreCrowd = nn.Parameter(torch.Tensor(self.a_num, self.tagset_size, self.tagset_size))

        # hidden2tag layer
        if 'maCatVecCrowd' in self.task:
            self.hidden2tag = nn.Linear(hidden_dim+self.tagset_size, self.tagset_size, bias=if_bias)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)

        if not (('maMulCRFCrowd' in self.task) and ('latent' not in self.task)):
            self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

        # attention
        self.attention = nn.Parameter(torch.Tensor(self.hidden_dim, self.a_num))
        self.softmax = nn.Softmax(dim=1)

    def rand_init(self):
        """
        random initialization
        """
        if 'maMulVecCrowd' in self.task:
            self.maMulVecCrowd.data.fill_(1)
        if 'maAddVecCrowd' in self.task:
            self.maAddVecCrowd.data.fill_(0)
        if 'maCatVecCrowd' in self.task:
            self.maCatVecCrowd.data.fill_(0)
            self.maCatVecCrowd_latent.data.fill_(0)
        if 'maMulMatCrowd' in self.task:
            for i in range(self.a_num):
                nn.init.eye_(self.maMulMatCrowd[i])
        if 'maMulCRFCrowd' in self.task:
            for i in range(self.a_num):
                nn.init.eye_(self.maMulCRFCrowd[i])
        if 'maMulScoreCrowd' in self.task:
            for i in range(self.a_num):
                nn.init.eye_(self.maMulScoreCrowd[i])

        utils.init_linear(self.hidden2tag)
        self.attention.data.zero_()

        if not (('maMulCRFCrowd' in self.task) and ('latent' not in self.task)):
            self.transitions.data.zero_()
        

    def forward(self, feats, aid):
        """
        calculate the potential score for the conditional random field.

        Parameters
        ----------
        feats: ``torch.FloatTensor``, required.
            the input features for the conditional random field, of shape (*, hidden_dim).

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (ins_num, from_tag_size, to_tag_size)
        """

        seq_len, batch_size, hid_dim = feats.shape
        ins_num = seq_len * batch_size
        ains_num = self.a_num * ins_num

        # attention
        feats_tmp = feats.view(seq_len, batch_size, 2, self.hidden_dim // 2)
        snt_emb = torch.cat([feats_tmp[-1,:,0,:], feats_tmp[0,:,1,:]], 1)
        att = torch.matmul(snt_emb, self.attention)
        att = self.softmax(att)

        # ner
        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        transitions = self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + transitions

        if 'maMulScoreCrowd' in self.task:
            crowd = self.maMulScoreCrowd.view(self.a_num, -1)
            if aid < 0:

                if self.test_att:
                    ## use aggregated crowd
                    crowd = torch.matmul(att, crowd)
                    crowd = crowd.view(1, batch_size, self.tagset_size, self.tagset_size)
                    crowd = crowd.expand(seq_len, batch_size, self.tagset_size, self.tagset_size).contiguous().view(ins_num, self.tagset_size, self.tagset_size)

                    crf_scores = torch.matmul(crf_scores, crowd).view(ins_num, self.tagset_size, self.tagset_size)
                else:
                    ## only use latent
                    crf_scores = crf_scores.view(ins_num, self.tagset_size, self.tagset_size)
            else:
                if self.train_att:
                    # use aggregated crowd
                    crowd = torch.matmul(att, crowd)
                    crowd = crowd.view(1, batch_size, self.tagset_size, self.tagset_size)
                else:
                    # use indicing crowd
                    crowd = crowd[aid]
                    crowd = crowd.view(1, 1, self.tagset_size, self.tagset_size)

                crowd = crowd.expand(seq_len, batch_size, self.tagset_size, self.tagset_size).contiguous().view(ins_num, self.tagset_size, self.tagset_size)
                crf_scores = torch.matmul(crf_scores, crowd).view(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores, att
        

        """
        # hidden2tag
        if 'maCatVecCrowd' in self.task:
            feats_expand = feats.expand(self.a_num, seq_len, batch_size, hid_dim)
            crowd_expand = self.maCatVecCrowd.unsqueeze(1).unsqueeze(2).expand(self.a_num, seq_len, batch_size, self.tagset_size)
            feats_cat = torch.cat([feats_expand, crowd_expand], 3)
            scores = self.hidden2tag(feats_cat).view(self.a_num, ins_num, 1, self.tagset_size)
        else:
            scores = self.hidden2tag(feats).view(1, ins_num, 1, self.tagset_size).expand(self.a_num, ins_num, 1, self.tagset_size)
        # scores : (self.a_num, ins_num, 1, self.tagset_size)

        # transition on tag score
        if 'maMulVecCrowd' in self.task:
            crowd_expand = self.maMulVecCrowd.unsqueeze(1).unsqueeze(2).expand(self.a_num, ins_num, 1, self.tagset_size)
            scores = torch.mul(scores, crowd_expand)
        if 'maAddVecCrowd' in self.task:
            crowd_expand = self.maAddVecCrowd.unsqueeze(1).unsqueeze(2).expand(self.a_num, ins_num, 1, self.tagset_size)
            scores = scores + crowd_expand
        if 'maMulMatCrowd' in self.task:
            crowd = F.log_softmax(self.maMulMatCrowd, dim=2)
            crowd = crowd.view(self.a_num, 1, self.tagset_size, self.tagset_size).expand(self.a_num, ins_num, self.tagset_size, self.tagset_size)
            scores = torch.matmul(scores, crowd)#.transpose(0,1).contiguous()
        # scores : (self.a_num, ins_num, 1, self.tagset_size)

        # transition matrix
        if 'maMulCRFCrowd' in self.task and 'latent' in self.task:
            transitions = self.transitions.view(1, 1, self.tagset_size, self.tagset_size)
            transitions = torch.matmul(transitions, self.maMulCRFCrowd).transpose(0, 1).contiguous() 
            #transitions = F.log_softmax(transitions, dim=2)
        elif 'maMulCRFCrowd' in self.task:
            transitions = self.maMulCRFCrowd.view(self.a_num, 1, self.tagset_size, self.tagset_size).expand(self.a_num, ins_num, self.tagset_size, self.tagset_size)
        else:
            transitions = self.transitions.view(1, 1, self.tagset_size, self.tagset_size).expand(self.a_num, ins_num, self.tagset_size, self.tagset_size)
        # transitions: (self.a_num, ins_num, self.tagset_size, self.tagset_size)

        scores = scores.expand(self.a_num, ins_num, self.tagset_size, self.tagset_size)
        crf_scores = scores + transitions
        # crf_scores: (self.a_num, ins_num, self.tagset_size, self.tagset_size)

        # score transformation
        if 'maMulScoreCrowd' in self.task:
            crowd = self.maMulScoreCrowd.view(self.a_num, 1, self.tagset_size, self.tagset_size).expand(self.a_num, ins_num, self.tagset_size, self.tagset_size)
            crf_scores = torch.matmul(crf_scores, crowd).view(self.a_num, ins_num, self.tagset_size, self.tagset_size)

        return crf_scores
        """

    def latent_forward(self, feats):
        """
        ignoring crowd components
        """
        seq_len, batch_size, hid_dim = feats.shape
        if 'maCatVecCrowd' in self.task:
            crowd_zero = self.maCatVecCrowd_latent.view(1, 1, self.tagset_size).expand(seq_len, batch_size, self.tagset_size)
            feats = torch.cat([feats, crowd_zero], 2)

        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores
        

class CRFLoss(nn.Module):
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    average_batch : ``bool``, optional, (default=True).
        whether the return score would be averaged per batch.
    """

    def __init__(self, 
                 y_map: dict, 
                 average_batch: bool = True):
        super(CRFLoss, self).__init__()
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        calculate the negative log likehood for the conditional random field.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (seq_len, batch_size, from_tag_size, to_tag_size).
        target: ``torch.LongTensor``, required.
            the positive path for the conditional random field, of shape (seq_len, batch_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).

        Returns
        -------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target.unsqueeze(2)).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()

        seq_iter = enumerate(scores)

        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.start_tag, :].squeeze(1).clone()

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.unsqueeze(2).expand(bat_size, self.tagset_size, self.tagset_size)

            cur_partition = utils.log_sum_exp(cur_values)

            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))

        partition = partition[:, self.end_tag].sum()

        if self.average_batch:
            return (partition - tg_energy) / bat_size
        else:
            return (partition - tg_energy)
        #return (partition, tg_energy)

class CRFLoss_ma(nn.Module):
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    average_batch : ``bool``, optional, (default=True).
        whether the return score would be averaged per batch.
    antor_num: ``int``, required
            the number of annotators
    task: ``string``. required
            the task: [maWeightAnnotator, maWeightLabel, maWeightAnnoLabel]
    """

    def __init__(self, 
                 y_map: dict, 
                 task: str,
                 a_num: int = 1,
                 average_batch: bool = True):
        super(CRFLoss_ma, self).__init__()
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.unk_tag = y_map['<unk>']
        self.average_batch = average_batch
        self.a_num = a_num
        self.task = task

    def forward(self, scores, targets, mask, a_mask):
        """
        calculate the negative log likehood for the conditional random field.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (a_num, seq_len, batch_size, from_tag_size, to_tag_size).
        targets: ``torch.LongTensor``, required.
            the positive path for the conditional random field, of shape (a_num, seq_len, batch_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).
        a_mask: ``torch.ByteTensor``, required.
            the mask for the valid annotator, of shape (a_num, batch_size)

        Returns
        -------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        a_num = scores.size(0)
        seq_len = scores.size(1)
        bat_size = scores.size(2)

        mask_not = mask == 0 
        a_mask_not = a_mask == 0 

        losses = a_mask.clone().float()

        for aid in range(a_num):
            target = targets[aid]
            score = scores[aid]

            #print('score',score.size())
            #print('target',target.size())
            tg_energy = torch.gather(score.view(seq_len, bat_size, -1), 2, target.unsqueeze(2)).view(seq_len, bat_size)
            #tg_energy = tg_energy.masked_select(mask).sum()
            tg_energy = tg_energy.masked_fill_(mask_not, 0).sum(dim=0)

            seq_iter = enumerate(score)

            _, inivalues = seq_iter.__next__()
            partition = inivalues[:, self.start_tag, :].squeeze(1).clone()

            for idx, cur_values in seq_iter:
                cur_values = cur_values + partition.unsqueeze(2).expand(bat_size, self.tagset_size, self.tagset_size)

                cur_partition = utils.log_sum_exp(cur_values)

                mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
                partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))

            partition = partition[:, self.end_tag] # [bat_size]
            losses[aid] = partition - tg_energy
        #print('losses',losses)

        loss = losses.masked_select(a_mask).sum()
        #print('a_mask',a_mask)
        #print('loss',loss)

        if self.average_batch:
            return loss / bat_size
        else:
            return loss

    def to_params(self):
        """
        To parameters.
        """
        if self.task == 'maSumLoss':
            return {
                "model_type": "crf-loss",
                "task": self.task
            }
        elif self.task == 'maWeightAnnotator':
            return {
                "model_type": "crf-loss",
                "task": self.task,
                "antor_score": self.antor_score
            }
        elif self.task == 'maWeightLabel':
            return {
                "model_type": "crf-loss",
                "task": self.task,
                "label_score": self.label_score
            }
        elif self.task == 'maWeightAnnoLabel':
            return {
                "model_type": "crf-loss",
                "task": self.task,
                "antor_score": self.antor_score,
                "label_score": self.label_score
            }

class CRFLoss_ma_mturk(nn.Module):
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    average_batch : ``bool``, optional, (default=True).
        whether the return score would be averaged per batch.
    """

    def __init__(self, 
                 y_map: dict, 
                 task: str,
                 antor_num: int = 1,
                 average_batch: bool = True):
        super(CRFLoss_ma_mturk, self).__init__()
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.unk_tag = y_map['<unk>']
        self.average_batch = average_batch
        self.antor_num = antor_num
        self.task = task

        if self.task == 'maWeightAnnotator': 
            self.antor_score = nn.Parameter(torch.rand(self.antor_num))

    def forward(self, scores, target, mask, aid):
        """
        calculate the negative log likehood for the conditional random field.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (seq_len, batch_size, from_tag_size, to_tag_size).
        target: ``torch.LongTensor``, required.
            the positive path for the conditional random field, of shape (seq_len, batch_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).

        Returns
        -------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target.unsqueeze(2)).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()

        seq_iter = enumerate(scores)

        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.start_tag, :].squeeze(1).clone()

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.unsqueeze(2).expand(bat_size, self.tagset_size, self.tagset_size)

            cur_partition = utils.log_sum_exp(cur_values)

            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))

        partition = partition[:, self.end_tag].sum()

        loss = partition - tg_energy

        if self.task == 'maWeightAnnotator':  
            antor_score = F.softmax(self.antor_score, dim=0)
            loss = loss * antor_score[aid] 

        if self.average_batch:
            return loss / bat_size
        else:
            return loss

    def to_params(self):
        """
        To parameters.
        """
        if self.task == 'maSumLoss':
            return {
                "model_type": "crf-loss",
                "task": self.task
            }
        elif self.task == 'maWeightAnnotator':
            return {
                "model_type": "crf-loss",
                "task": self.task,
                "antor_score": self.antor_score
            }
        elif self.task == 'maWeightLabel':
            return {
                "model_type": "crf-loss",
                "task": self.task,
                "label_score": self.label_score
            }
        elif self.task == 'maWeightAnnoLabel':
            return {
                "model_type": "crf-loss",
                "task": self.task,
                "antor_score": self.antor_score,
                "label_score": self.label_score
            }

class CRFDecode():
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    """
    def __init__(self, y_map: dict):
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.y_map = y_map
        self.r_y_map = {v:k for k, v in self.y_map.items()}

    def decode(self, scores, mask):
        """
        find the best path from the potential scores by the viterbi decoding algorithm.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (seq_len, batch_size, from_tag_size, to_tag_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).

        Returns
        -------
        output: ``torch.LongTensor``.
            A LongTensor of shape (seq_len - 1, batch_size)
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask.data
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        forscores = inivalues[:, self.start_tag, :]
        back_points = list()

        for idx, cur_values in seq_iter:
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            back_point = back_points[idx]
            index = pointer.contiguous().view(-1, 1)
            pointer = torch.gather(back_point, 1, index).view(-1)
            decode_idx[idx] = pointer
        return decode_idx

    def to_spans(self, sequence):
        """
        decode the best path to spans.

        Parameters
        ----------
        sequence: list, required.
            the list of best label indexes paths .

        Returns
        -------
        output: ``set``.
            A set of chunks contains the position and type of the entities.
        """
        chunks = []
        current = None

        for i, y in enumerate(sequence):
            label = self.r_y_map[y]

            if label.startswith('B-'):

                if current is not None:
                    chunks.append('@'.join(current))
                current = [label.replace('B-', ''), '%d' % i]

            elif label.startswith('S-'):

                if current is not None:
                    chunks.append('@'.join(current))
                    current = None
                base = label.replace('S-', '')
                chunks.append('@'.join([base, '%d' % i]))

            elif label.startswith('I-'):

                if current is not None:
                    base = label.replace('I-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]

                else:
                    current = [label.replace('I-', ''), '%d' % i]

            elif label.startswith('E-'):

                if current is not None:
                    base = label.replace('E-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                        chunks.append('@'.join(current))
                        current = None
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]
                        chunks.append('@'.join(current))
                        current = None

                else:
                    current = [label.replace('E-', ''), '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                if current is not None:
                    chunks.append('@'.join(current))
                current = None

        if current is not None:
            chunks.append('@'.join(current))

        return set(chunks)
