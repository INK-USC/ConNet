import numpy as np
import torch


# pack data for the model
def pack_data(args, data):
    packed_data = []
    for i in range(int(np.ceil(len(data) / args['batch']))):
        batch = data[(i*args['batch']):((i+1)*args['batch'])]
        
        # data id (save for error analysis)
        data_ids = [r['data_id'] for r in batch]
        
        # data input: chars and/or words
        data_input = [None, None]
        char_max_len = max([len(d['chars']) for d in batch])
        # adding 1 to word_max_len to ensure every sentence and label sequence ends with "<pad>"
        # this is for working with CRF. CRF should have a better implementation without this requirement
        word_max_len = max([len(d['words']) for d in batch]) + 1
        if args['char_rnn']:
            chars = [d['chars']+[args['pad_char_idx']]*(char_max_len-len(d['chars'])) for d in batch]
            chars = torch.LongTensor(np.array(chars))
            if args['cuda']:
                chars = chars.cuda()
            char_boundaries = [d['char_boundaries']+[[char_max_len-1, char_max_len-1]]*(word_max_len-len(d['char_boundaries'])) for d in batch]
            char_start = [[r[0] for r in rr] for rr in char_boundaries]
            char_end = [[r[1] for r in rr] for rr in char_boundaries]
            data_input[0] = [chars, char_start, char_end]
        if args['word_rnn']:
            words = [d['words']+[args['pad_word_idx']]*(word_max_len-len(d['words'])) for d in batch]
            words = torch.LongTensor(np.array(words))
            if args['cuda']:
                words = words.cuda()
            data_input[1] = words
        
        # label
        label = [[args['start_label_idx']]+d['label']+[args['pad_label_idx']]*(word_max_len-len(d['label'])) for d in batch]
        label = [[r[i]*args['tagset_size']+r[i+1] for i in range(len(r)-1)] for r in label]
        label = torch.LongTensor(np.array(label))
        if args['cuda']:
            label = label.cuda()
        
        # other information
        task_ids = [r['task'] for r in batch]
        sent_end = [len(d['words']) for d in batch]
        mask = (label != args['pad_label_idx']*(args['tagset_size']+1))
        
        packed_data.append({'data_ids': data_ids, 'data_input': data_input, 'label': label, \
                            'task_ids': task_ids, 'sent_end': sent_end, 'mask': mask})
    
    return packed_data

# pack data based on the tasks (each batch contains only one task)
def pack_data_tasks(args, data, task2idx):
    packed_data = []
    for task, id in task2idx.items():
        data_task = [r for r in data if r['task']==id]
        packed_data += pack_data(args, data_task)
    
    return packed_data

# pack data for MLP models
def pack_data_mlp(args, data):
    packed_data = []
    for i in range(int(np.ceil(len(data) / args['batch']))):
        batch = data[(i*args['batch']):((i+1)*args['batch'])]
        
        # data id (save for error analysis)
        data_ids = [r['data_id'] for r in batch]
        
        # data input: word features
        data_input = torch.FloatTensor([d['word_feat'] for d in batch])
        if args['cuda']:
                data_input = data_input.cuda()
        
        # label
        label = [d['label'] for d in batch]
        label = torch.FloatTensor(np.array(label))
        if args['cuda']:
            label = label.cuda()
        
        # other information
        task_ids = [r['task'] for r in batch]
        
        packed_data.append({'data_ids': data_ids, 'data_input': data_input, 'label': label, \
                            'task_ids': task_ids})
    
    return packed_data