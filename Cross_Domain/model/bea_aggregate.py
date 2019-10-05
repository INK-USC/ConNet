import os
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.preprocessing import LabelEncoder


def get_entities(labels):
    entities = []
    pre_label = 'O'
    pre_tag = 'O'
    pre_sep = 0
    for cur_idx, cur_label in enumerate(np.append(labels, 'O')):
        cur_tag = cur_label.split('-')[-1]
        
        if cur_tag != pre_tag or cur_label.startswith('B-'):
            if pre_tag != 'O':
                entities.append(((pre_sep, cur_idx), pre_tag))
            
            pre_sep = cur_idx
        
        pre_label = cur_label
        pre_tag = cur_tag
    
    return entities

def get_f1(s1, s2):
    return 2*len(s1 & s2) / (len(s1) + len(s2)) * 100

def mv_infer(values):
    num_classes = values.max() + 1
    num_items, num_workers = values.shape
    
    all_items = np.arange(num_items)
    z_ik = np.zeros((num_items, num_classes))
    
    for j in range(num_workers):
        z_ik[all_items, values[:, j]] += 1
    
    return z_ik

def get_Eq_log_pi_k_and_Eq_log_v_jkl(values, z_ik, alpha_k=1, beta_kl=1):
    num_items, num_workers = values.shape
    num_classes = z_ik.shape[1]
    
    alpha_k = alpha_k * np.ones(num_classes)
    beta_kl = beta_kl * np.ones((num_classes, num_classes))
    
    Eq_log_pi_k = digamma(z_ik.sum(axis=0) + alpha_k) - digamma(num_items + alpha_k.sum())
    
    n_jkl = np.zeros((num_workers, num_classes, num_classes)) + beta_kl
    for j in range(num_workers):
        for k in range(num_classes):
            n_jkl[j, k, :] += np.bincount(values[:, j], z_ik[:, k], minlength=num_classes)
    
    Eq_log_v_jkl = digamma(n_jkl) - digamma(n_jkl.sum(axis=-1, keepdims=True))
    
    return Eq_log_pi_k, Eq_log_v_jkl

def get_z_ik(values, Eq_log_v_jkl, Eq_log_pi_k=None, prior=False):
    num_items, num_workers = values.shape
    num_classes = Eq_log_v_jkl.shape[1]
    
    z_ik = np.zeros((num_items, num_classes))
    if prior:
        z_ik += Eq_log_pi_k
    
    for j in range(num_workers):
        z_ik += Eq_log_v_jkl[j, :, values[:, j]]
    
    z_ik -= z_ik.max(axis=-1, keepdims=True)
    z_ik = np.exp(z_ik)
    z_ik /= z_ik.sum(axis=-1, keepdims=True)
    
    return z_ik

def bea_infer_wiki(values, alpha_k=1, beta_kl=1, prior=True):
    z_ik = mv_infer(values)
    for iteration in range(500):
        Eq_log_pi_k, Eq_log_v_jkl = get_Eq_log_pi_k_and_Eq_log_v_jkl(values, z_ik, alpha_k, beta_kl)
        
        last_z_ik = z_ik
        z_ik = get_z_ik(values, Eq_log_v_jkl, Eq_log_pi_k, prior)
        
        if np.allclose(z_ik, last_z_ik, atol=1e-3):
            break
        
    return z_ik, Eq_log_v_jkl, Eq_log_pi_k, iteration

def bea_infer_conll(values, alpha=1, beta_kl=1, prior=True):
    num_classes = values.max() + 1
    num_items, num_workers = values.shape
    
    beta_kl = beta_kl * np.ones((num_classes, num_classes))
    
    z_ik = mv_infer(values)
    n_jkl = np.empty((num_workers, num_classes, num_classes))
    
    last_z_ik = z_ik.copy()
    for iteration in range(500):
        Eq_log_pi_k = digamma(z_ik.sum(axis=0) + alpha) - digamma(num_items + num_classes*alpha)
        
        n_jkl[:] = beta_kl
        for j in range(num_workers):
            for k in range(num_classes):
                n_jkl[j, k, :] += np.bincount(values[:, j], z_ik[:, k], minlength=num_classes)        
        
        Eq_log_v_jkl = digamma(n_jkl) - digamma(n_jkl.sum(axis=-1, keepdims=True))
        
        if prior:
            z_ik[:] = Eq_log_pi_k
        else:
            z_ik.fill(0)
        
        for j in range(num_workers):
            z_ik += Eq_log_v_jkl[j, :, values[:, j]]
        z_ik -= z_ik.max(axis=-1, keepdims=True)
        z_ik = np.exp(z_ik)
        z_ik /= z_ik.sum(axis=-1, keepdims=True)
        
        if np.allclose(z_ik, last_z_ik, atol=1e-3):
            break
        
        last_z_ik[:] = z_ik
    return z_ik, iteration

def get_entities_from_ent_results(z_ik, df_range):
    df = pd.DataFrame(z_ik, index=df_range.index.set_names(['beg', 'end']), columns=pd.Series(tag_le.classes_, name='tag'))
    df = df.stack().rename('prob').reset_index().sort_values('prob', ascending=False).drop_duplicates(['beg', 'end'])
    num_items = df.end.max()
    df = df[df['tag'] != 'O']
    
    pred_entities = set()
    occupied = np.zeros(num_items)
    for beg, end, tag, prob in df.values:
        if occupied[beg:end].sum() == 0:
            occupied[beg:end] = 1
            pred_entities.add(((beg, end), tag))
        
    return pred_entities

def bea_ent(df_range, **kwargs):
    z_ik, Eq_log_v_jkl, Eq_log_pi_k, iteration = bea_infer_wiki(df_range.values, **kwargs)
    return get_entities_from_ent_results(z_ik, df_range), Eq_log_v_jkl, Eq_log_pi_k, iteration

def get_df_range(df_label):
    return pd.DataFrame({source: dict(get_entities(label_le.inverse_transform(df_label[source].values)))
                         for source in df_label.columns}).fillna('O').apply(tag_le.transform)

def get_df_recall(Eq_log_v_jkl, sources):
    v_jkl = np.exp(Eq_log_v_jkl)
    v_jkl /= v_jkl.sum(axis=-1, keepdims=True)
    
    df_recall = pd.DataFrame(v_jkl[:, np.arange(num_tags), np.arange(num_tags)], columns=tag_le.classes_)
    df_recall['source'] = sources
    df_recall['Avg3'] = df_recall[['LOC', 'ORG', 'PER']].mean(axis=1)
    
    return df_recall


def bea_ent_x2_t10(true, pred, all_labels, all_tags):
    global label_le, tag_le, num_classes, num_tags
    label_le = LabelEncoder().fit(all_labels)
    tag_le = LabelEncoder().fit(all_tags)
    num_classes = len(label_le.classes_)
    num_tags = len(tag_le.classes_)
    
    a_v, b_v = 1, 1
    beta_kl = np.eye(num_classes) * (a_v-b_v) + b_v
    beta_kl_tag = np.eye(num_tags) * (a_v-b_v) + b_v
    
    pred = [label_le.transform(r) for r in pred]
    df_label = pd.DataFrame(np.array(pred).transpose())
    true_entities = set(get_entities(true))
    df_range = get_df_range(df_label)
    _, Eq_log_v_jkl = bea_ent(df_range, beta_kl=beta_kl_tag, prior=True)[:2]
    
    # spammer removel
    # round 1, pick top 20
    df_recall = get_df_recall(Eq_log_v_jkl, df_range.columns).sort_values('Avg3', ascending=False)
    df_range = get_df_range(df_label[df_recall.source[:20]])
    _, Eq_log_v_jkl = bea_ent(df_range, beta_kl=beta_kl_tag, prior=True)[:2]
    
    # round 2, pick top 10
    df_recall = get_df_recall(Eq_log_v_jkl, df_range.columns).sort_values('Avg3', ascending=False)
    
    df_range = get_df_range(df_label[df_recall.source[:10]])
    pred_entities, Eq_log_v_jkl = bea_ent(df_range, beta_kl=beta_kl_tag, prior=True)[:2]
    f1 = get_f1(true_entities, pred_entities)
    pred_seq = ['O' for r in range(len(true))]
    for (start, end), tag in list(pred_entities):
        pred_seq[start] = 'B-'+tag if '<' not in tag else tag
        for i in range(start+1,end):
            pred_seq[i] = 'I-'+tag if '<' not in tag else tag
    
    return pred_seq, f1


def bea_ent_uns(true, pred, all_labels, all_tags, ner=True):
    label_le = LabelEncoder().fit(all_labels)
    tag_le = LabelEncoder().fit(all_tags)
    num_classes = len(label_le.classes_)
    num_tags = len(tag_le.classes_)
    
    a_v, b_v = 1, 1
    beta_kl = np.eye(num_classes) * (a_v-b_v) + b_v
    beta_kl_tag = np.eye(num_tags) * (a_v-b_v) + b_v
    
    if not ner:
        true = ['B-'+r for r in true]
        pred = [['B-'+rr for rr in r] for r in pred]
    
    df_label = pd.DataFrame(np.array(pred).transpose())
    true_entities = set(get_entities(true))
    
    df_range = pd.DataFrame({source: dict(get_entities(df_label[source].values))
                         for source in df_label.columns}).fillna('O')
    values_range = np.column_stack([tag_le.transform(df_range[source]) for source in df_range.columns])
    z_ik, iteration = bea_infer_conll(values_range, beta_kl=beta_kl_tag, prior=True)
    pred_entities = set([(rng, tag) for (rng, tag) 
                         in zip(df_range.index.values, tag_le.inverse_transform(z_ik.argmax(axis=-1))) if tag != 'O'])
    
    f1 = get_f1(true_entities, pred_entities)
    pred_seq = ['O' for r in range(len(true))]
    for (start, end), tag in list(pred_entities):
        if not ner:
            pred_seq[start] = tag
            assert start+1 == end
        else:
            pred_seq[start] = 'B-'+tag if '<' not in tag else tag
            for i in range(start+1,end):
                pred_seq[i] = 'I-'+tag if '<' not in tag else tag
    
    return pred_seq, f1


if __name__ == '__main__':
    all_labels = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']
    all_tags = ['LOC', 'ORG', 'PER', 'O']
    
    true = [ 'O', 'B-LOC', 'I-LOC', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER']
    pred = [['O', 'B-LOC', 'I-LOC', 'B-PER', 'O', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER'], \
            ['O', 'B-LOC', 'I-LOC', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'O'    , 'B-PER'], \
            ['O', 'O'    , 'B-LOC', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER'], \
            ['O', 'B-PER', 'O'    , 'B-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER']]
    
    # less than 10 annotators
    pred_seq, f1 = bea_ent_uns(true, pred, all_labels, all_tags)
    print('bea_ent_uns:', f1)
    print(pred_seq)
    
    # more than 10 annotators
    pred = [pred[0]]*10 + [pred[1]]*20 + [pred[2]]*30 + [pred[3]]*40
    pred_seq, f1 = bea_ent_x2_t10(true, pred, all_labels, all_tags)
    print('bea_ent_unsx2_t10:', f1)
    print(pred_seq)
    
    # POS tagging
    all_labels = ['ABC', 'DEF', 'HIJ', 'KLM', 'OPQ']
    all_tags = ['ABC', 'DEF', 'HIJ', 'KLM', 'OPQ']
    
    true = [ 'ABC', 'DEF', 'HIJ', 'KLM', 'OPQ', 'ABC', 'DEF', 'HIJ', 'KLM', 'OPQ']
    pred = [['ABC', 'ABC', 'HIJ', 'KLM', 'OPQ', 'ABC', 'DEF', 'HIJ', 'DEF', 'OPQ'], \
            ['ABC', 'DEF', 'ABC', 'KLM', 'OPQ', 'ABC', 'DEF', 'DEF', 'KLM', 'DEF'], \
            ['ABC', 'DEF', 'HIJ', 'ABC', 'OPQ', 'DEF', 'DEF', 'HIJ', 'KLM', 'OPQ'], \
            ['ABC', 'DEF', 'HIJ', 'KLM', 'ABC', 'ABC', 'DEF', 'HIJ', 'KLM', 'OPQ']]
    
    pred_seq, f1 = bea_ent_uns(true, pred, all_labels, all_tags, False)
    print('bea_ent_uns:', f1)
    print(pred_seq)