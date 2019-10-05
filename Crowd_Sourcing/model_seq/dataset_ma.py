"""
.. module:: dataset
    :synopsis: dataset for sequence labeling
 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import pickle
import random
import functools
import itertools
from tqdm import tqdm

class SeqDatasetMA(object):
    """    
    Dataset for Sequence Labeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_con : ``int``, required.
        The index of connect character token for character-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    y_start : ``int``, required.
        The index of the start label token.
    y_pad : ``int``, required.
        The index of the pad label token.
    y_size : ``int``, required.
        The size of the tag set.
    batch_size: ``int``, required.
        Batch size.
    """
    def __init__(self, 
                dataset: list, 
                w_pad: int, 
                c_con: int, 
                c_pad: int, 
                y_start: int, 
                y_pad: int, 
                y_size: int, 
                batch_size: int,
                y_unk: int = -1,
                y_O: int = -1,
                set_unk: bool = False,
                set_O: bool = False,
                set_unchange: bool = False,
                if_shuffle: bool = True): 
        super(SeqDatasetMA, self).__init__()

        self.w_pad = w_pad
        self.c_con = c_con
        self.c_pad = c_pad
        self.y_pad = y_pad
        self.y_unk = y_unk
        self.y_size = y_size
        self.y_start = y_start
        self.y_O = y_O
        self.batch_size = batch_size
        self.set_unk = set_unk
        self.set_O = set_O
        self.set_unchange = set_unchange
        self.if_shuffle = if_shuffle

        self.construct_index(dataset)
        if self.if_shuffle:
            self.shuffle()

    def shuffle(self):
        """
        shuffle dataset
        """
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def construct_index(self, dataset):
        """
        construct index for the dataset.

        Parameters
        ----------
        dataset: ``list``, required.
            the encoded dataset (outputs of preprocess scripts).        
        """
        for instance in dataset:
            """
            instance: [words : [], characters : [[]], labels: [[]]]
                words: seq_len
                characters: seq_len x word_len -> [] : char_len + (seq_len - 1)
                labels: seq_len x antor_num
               +c_len
            """
            c_len = [len(tup)+1 for tup in instance[1]]
            c_ins = [tup for ins in instance[1] for tup in (ins + [self.c_con])]
            instance[1] = c_ins
            instance.append(c_len)

        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))
    
    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if self.if_shuffle:
            self.shuffle()
    
    def batchify(self, batch, device):
        """
        batchify a batch of data and move to a device.

        Parameters
        ----------
        batch: ``list``, required.
            a sample from the encoded dataset (outputs of preprocess scripts).  
        device: ``torch.device``, required.
            the target device for the dataset loader.
        """
        
        cur_batch_size = len(batch)

        char_padded_len = max([len(tup[1]) for tup in batch])
        word_padded_len = max([len(tup[0]) for tup in batch])

        tmp_batch =  [list() for ind in range(9)]

        for instance_ind in range(cur_batch_size):

            instance = batch[instance_ind] 

            """
            instance[0]: a list of words
                    [1]: a list of chars
                    [2]: a list of tags
                    [3]: a list of word_lens

            tmp_batch[0]: f_c, forward character
                     [1]: f_p, forward char pos
                     [2]: b_c, back char
                     [3]: b_p, back char pos
                     [4]: f_w, forw padded word
                     [5]: f_y, forw padded tags (multi-annotator)
                     [6]: f_y_m, forw tag mask
                     [7]: a_m, valid annotator mask
                     [8]: g_y, a list of tags
            """

            char_padded_len_ins = char_padded_len - len(instance[1])
            word_padded_len_ins = word_padded_len - len(instance[0])

            tmp_batch[0].append(instance[1] + [self.c_pad] + [self.c_pad] * char_padded_len_ins)
            tmp_batch[2].append([self.c_pad] + instance[1][::-1] + [self.c_pad] * char_padded_len_ins)

            tmp_p = list( itertools.accumulate(instance[3]+[1]+[0]* word_padded_len_ins) )
            tmp_batch[1].append([(x - 1) * cur_batch_size + instance_ind for x in tmp_p])
            tmp_p = list(itertools.accumulate([1]+instance[3][::-1]))[::-1] + [1]*word_padded_len_ins
            tmp_batch[3].append([(x - 1) * cur_batch_size + instance_ind for x in tmp_p])

            tmp_batch[4].append(instance[0] + [self.w_pad] + [self.w_pad] * word_padded_len_ins)

            # deal with tags
            a_num = len(instance[2][0])
            labels = []
            for i in range(a_num):
                label = [y[i] for y in instance[2]]
                labels.append([self.y_start * self.y_size + label[0]] + [label[ind] * self.y_size + label[ind+1] for ind in range(len(label) - 1)] + [label[-1] * self.y_size + self.y_pad] + [self.y_pad * self.y_size + self.y_pad] * word_padded_len_ins)
            tmp_batch[5].append(labels)

            tmp_batch[6].append([1] * len(instance[2]) + [1] + [0] * word_padded_len_ins)

            # valid annotator
            """
            vals = []
            for i in range(a_num):
                val = 0
                for y in instance[2]:
                    if y[i] != self.y_unk:
                        val = 1
                vals.append(val)
            tmp_batch[7].append(vals)
            """
            vals = []
            for i in range(a_num):
                val = False
                for y in instance[2]:
                    if y[i] != self.y_unk:
                        val = True
                        break
                if val:
                    #vals.append([1] * len(instance[2]) + [1] + [0] * word_padded_len_ins)
                    vals.append(1)
                else:
                    #vals.append([0] * (len(instance[2]) + 1 + word_padded_len_ins))
                    vals.append(0)
            tmp_batch[7].append(vals)

            tmp_batch[8].append(instance[2]) # labels


                
        tbt = [torch.LongTensor(v).transpose(0, 1).contiguous() for v in tmp_batch[0:6]] + [torch.ByteTensor(v).transpose(0, 1).contiguous() for v in tmp_batch[6:8]]

        tbt[1] = tbt[1].view(-1)
        tbt[3] = tbt[3].view(-1)
        tbt[5] = tbt[5].transpose(1, 2).contiguous()
        #tbt[7] = tbt[7].transpose(1, 2).contiguous()

        return [ten.to(device) for ten in tbt] + [tmp_batch[8]]

class SeqDatasetMAY(object):
    """    
    Dataset for Sequence Labeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_con : ``int``, required.
        The index of connect character token for character-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    y_start : ``int``, required.
        The index of the start label token.
    y_pad : ``int``, required.
        The index of the pad label token.
    y_size : ``int``, required.
        The size of the tag set.
    batch_size: ``int``, required.
        Batch size.
    """
    def __init__(self, 
                dataset: list, 
                w_pad: int, 
                c_con: int, 
                c_pad: int, 
                y_start: int, 
                y_pad: int, 
                y_size: int, 
                batch_size: int,
                y_unk: int = -1,
                y_O: int = -1,
                set_unk: bool = False,
                set_O: bool = False,
                set_unchange: bool = False): 
        super(SeqDatasetMAY, self).__init__()

        self.w_pad = w_pad
        self.c_con = c_con
        self.c_pad = c_pad
        self.y_pad = y_pad
        self.y_unk = y_unk
        self.y_size = y_size
        self.y_start = y_start
        self.y_O = y_O
        self.batch_size = batch_size
        self.set_unk = set_unk
        self.set_O = set_O
        self.set_unchange = set_unchange

        self.construct_index(dataset)
        self.shuffle()

    def shuffle(self):
        """
        shuffle dataset
        """
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def construct_index(self, dataset):
        """
        construct index for the dataset.

        Parameters
        ----------
        dataset: ``list``, required.
            the encoded dataset (outputs of preprocess scripts).        
        """
        for instance in dataset:
            """
            instance: [words : [], characters : [[]], labels: [[]]]
                words: seq_len
                characters: seq_len x word_len -> [] : char_len + (seq_len - 1)
                labels: seq_len x antor_num
               +c_len
            """
            c_len = [len(tup)+1 for tup in instance[1]]
            c_ins = [tup for ins in instance[1] for tup in (ins + [self.c_con])]
            instance[1] = c_ins
            instance.append(c_len)

        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))
    
    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        self.shuffle()
    
    def batchify(self, batch, device):
        """
        batchify a batch of data and move to a device.

        Parameters
        ----------
        batch: ``list``, required.
            a sample from the encoded dataset (outputs of preprocess scripts).  
        device: ``torch.device``, required.
            the target device for the dataset loader.
        """
        
        cur_batch_size = len(batch)

        char_padded_len = max([len(tup[1]) for tup in batch])
        word_padded_len = max([len(tup[0]) for tup in batch])

        tmp_batch =  [list() for ind in range(10)]

        for instance_ind in range(cur_batch_size):

            instance = batch[instance_ind] 

            """
            instance[0]: a list of words
                    [1]: a list of chars
                    [2]: a list of tags
                    [3]: a list of word_lens

            tmp_batch[0]: f_c, forward character
                     [1]: f_p, forward char pos
                     [2]: b_c, back char
                     [3]: b_p, back char pos
                     [4]: f_w, forw padded word
                     [5]: f_y, forw crf padded tags (multi-annotator)
                     [6]: f_y_o, forw padded tags (multi-annotator)
                     [7]: f_y_m, forw tag mask
                     [8]: a_m, valid annotator mask
                     [9]: g_y, a list of tags
            """

            char_padded_len_ins = char_padded_len - len(instance[1])
            word_padded_len_ins = word_padded_len - len(instance[0])

            tmp_batch[0].append(instance[1] + [self.c_pad] + [self.c_pad] * char_padded_len_ins)
            tmp_batch[2].append([self.c_pad] + instance[1][::-1] + [self.c_pad] * char_padded_len_ins)

            tmp_p = list( itertools.accumulate(instance[3]+[1]+[0]* word_padded_len_ins) )
            tmp_batch[1].append([(x - 1) * cur_batch_size + instance_ind for x in tmp_p])
            tmp_p = list(itertools.accumulate([1]+instance[3][::-1]))[::-1] + [1]*word_padded_len_ins
            tmp_batch[3].append([(x - 1) * cur_batch_size + instance_ind for x in tmp_p])

            tmp_batch[4].append(instance[0] + [self.w_pad] + [self.w_pad] * word_padded_len_ins)

            # deal with tags
            a_num = len(instance[2][0])
            labels = []
            olabels = []
            for i in range(a_num):
                label = [y[i] for y in instance[2]]
                labels.append([self.y_start * self.y_size + label[0]] + [label[ind] * self.y_size + label[ind+1] for ind in range(len(label) - 1)] + [label[-1] * self.y_size + self.y_pad] + [self.y_pad * self.y_size + self.y_pad] * word_padded_len_ins)
                label = [self.y_pad if y[i] == self.y_unk else y[i] for y in instance[2]]
                olabels.append(label +[self.y_pad] + [self.y_pad] * word_padded_len_ins)
            tmp_batch[5].append(labels)
            tmp_batch[6].append(olabels)

            tmp_batch[7].append([1] * len(instance[2]) + [1] + [0] * word_padded_len_ins)

            # valid annotator
            """
            vals = []
            for i in range(a_num):
                val = 0
                for y in instance[2]:
                    if y[i] != self.y_unk:
                        val = 1
                vals.append(val)
            tmp_batch[7].append(vals)
            """
            vals = []
            for i in range(a_num):
                val = False
                for y in instance[2]:
                    if y[i] != self.y_unk:
                        val = True
                        break
                if val:
                    #vals.append([1] * len(instance[2]) + [1] + [0] * word_padded_len_ins)
                    vals.append(1)
                else:
                    #vals.append([0] * (len(instance[2]) + 1 + word_padded_len_ins))
                    vals.append(0)
            tmp_batch[8].append(vals)

            labels = []
            for i in range(a_num):
                label = [y[i] for y in instance[2]]
                labels.append(label)
            tmp_batch[9].append(labels) # labels

                
        tbt = [torch.LongTensor(v).transpose(0, 1).contiguous() for v in tmp_batch[0:7]] + [torch.ByteTensor(v).transpose(0, 1).contiguous() for v in tmp_batch[7:9]]

        tbt[1] = tbt[1].view(-1)
        tbt[3] = tbt[3].view(-1)
        tbt[5] = tbt[5].transpose(1, 2).contiguous()
        tbt[6] = tbt[6].transpose(1, 2).contiguous()
        #tbt[7] = tbt[7].transpose(1, 2).contiguous()

        return [ten.to(device) for ten in tbt] + [tmp_batch[9]]
