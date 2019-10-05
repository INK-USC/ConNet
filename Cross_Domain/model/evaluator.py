import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


class Binary_Evaluator:
    def __init__(self, args, predictor):
        self.args = args
        self.predictor = predictor
    
    def evaluate(self, data, model_, criterion, model, preds=None):
        if not preds:
            preds, avg_loss = self.predictor.predict(data, model_, criterion, model)
        else:
            avg_loss = np.Inf
        label = [r['label'] for r in data]
        
        acc = accuracy_score(label, preds)
        # return acc, prec, recall, f1, avg_loss
        return acc, np.Inf, np.Inf, np.Inf, avg_loss # prec, recall and f1 are not defined in token level evaluating


class Token_Evaluator:
    def __init__(self, args, predictor):
        self.args = args
        self.predictor = predictor
    
    def evaluate(self, data, model_, criterion, model, preds=None):
        if not preds:
            preds, avg_loss = self.predictor.predict(data, model_, criterion, model)
        else:
            avg_loss = np.Inf
        label = [r['label'] for r in data]
        preds = [preds[i][:len(label[i])] for i in range(len(label))]
        
        acc = accuracy_score([r for rr in label for r in rr], [r for rr in preds for r in rr])
        # return acc, prec, recall, f1, avg_loss
        return acc, np.Inf, np.Inf, np.Inf, avg_loss # prec, recall and f1 are not defined in token level evaluating


class Chunk_Evaluator:
    def __init__(self, args, predictor, label2idx):
        self.args = args
        self.predictor = predictor
        self.label2idx = label2idx
        if 'O' in self.label2idx:
            if any([r.startswith('E') for r in list(self.label2idx.keys())]):
                self.schema = 'iobes'
            else:
                self.schema = 'iob'
        else:
            if any([r.startswith('E') for r in list(self.label2idx.keys())]):
                self.schema = 'ibes'
            else:
                self.schema = 'ib'
        
        self.B = [v for k,v in self.label2idx.items() if k.startswith('B')]
        self.I = [v for k,v in self.label2idx.items() if k.startswith('I')]
        if self.schema in ['iobes', 'ibes']:
            self.E = [v for k,v in self.label2idx.items() if k.startswith('E')]
            self.S = [v for k,v in self.label2idx.items() if k.startswith('S')]
        if self.schema in ['iobes', 'iob']:
            self.O = [v for k,v in self.label2idx.items() if k.startswith('O')] + ['<start>']
    
    def iob_to_chunk(self, label):
        all_chunks = []
        for l in label:
            in_ent = False
            curr_chunks, curr_chunk = [], ()
            for i in range(len(l)+1):
                if i == len(l):
                    if in_ent:
                        curr_chunk += (i-1,)
                        curr_chunks.append(curr_chunk)
                        curr_chunk = ()
                        in_ent = False
                    break
                if l[i] in self.I:
                    continue
                elif l[i] in self.O:
                    if in_ent:
                        curr_chunk += (i-1,)
                        curr_chunks.append(curr_chunk)
                        curr_chunk = ()
                        in_ent = False
                elif l[i] in self.B:
                    if in_ent:
                        curr_chunk += (i-1,)
                        curr_chunks.append(curr_chunk)
                        curr_chunk = ()
                        in_ent = False
                    curr_chunk = (l[i], i)
                    in_ent = True
            all_chunks.append(curr_chunks)
        return all_chunks
    
    # need some modification before use
    def iobes_to_chunk(self, label):
        all_chunks = []
        for l in label:
            in_ent = False
            curr_chunks, curr_chunk = [], ()
            for i in range(len(l)):
                if l[i] in self.O + self.I:
                    continue
                elif l[i] in self.S:
                    if in_ent:
                        curr_chunk += (i-1,)
                        curr_chunks.append(curr_chunk)
                        curr_chunk = ()
                        in_ent = False
                    curr_chunks.append(('', i, i))
                elif l[i] in self.B:
                    if in_ent:
                        curr_chunk += (i-1,)
                        curr_chunks.append(curr_chunk)
                        curr_chunk = ()
                        in_ent = False
                    curr_chunk = ('', i)
                    in_ent = True
                elif l[i] in self.E:
                    if not in_ent:
                        continue
                    curr_chunk += (i,)
                    curr_chunks.append(curr_chunk)
                    curr_chunk = ()
                    in_ent = False
            all_chunks.append(curr_chunks)
        return all_chunks
    
    def chunk_evaluate(self, true_chunks, pred_chunks):
        num_pred, num_true, num_overlap = 0,0,0
        for i in range(len(true_chunks)):
            num_pred += len(pred_chunks[i])
            num_true += len(true_chunks[i])
            num_overlap += len(set(pred_chunks[i]).intersection(true_chunks[i]))
        
        prec = num_overlap / num_pred if num_pred != 0 else 0
        recall = num_overlap / num_true if num_true != 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if prec+recall != 0 else 0
        return prec, recall, f1
    
    def evaluate(self, data, model_, criterion, model, preds=None):
        if not preds:
            preds, avg_loss = self.predictor.predict(data, model_, criterion, model)
        else:
            avg_loss = np.Inf
        preds = [[r for r in rr if r != self.args['pad_label_idx']] for rr in preds]
        label = [r['label'] for r in data]
        if self.schema == 'iobes':
            true_chunks = self.iobes_to_chunk(label)
            pred_chunks = self.iobes_to_chunk(preds)
        elif self.schema == 'iob':
            true_chunks = self.iob_to_chunk(label)
            pred_chunks = self.iob_to_chunk(preds)
        prec, recall, f1 = self.chunk_evaluate(true_chunks, pred_chunks)
        return np.Inf, prec, recall, f1, avg_loss # accuracy is not defined in chunk level evaluating
    
