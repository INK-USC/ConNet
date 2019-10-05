import torch, os, pickle, sys
from model.models import *
from model.data_utils import *


for checkpoint_idx in range(6):
    checkpoint_dir = '/checkpoints/OntoNotes-5/ON_CN_2_with_agg_'+str(checkpoint_idx)+'.1'

    assert len(os.listdir(checkpoint_dir)) == 1

    saved = torch.load(checkpoint_dir + '/' + os.listdir(checkpoint_dir)[0])

    args = saved['args']

    # load data
    common = pickle.load(open(args['data_dir']+'/common.p', 'rb'))
    label2idx = common['label2idx']
    task2idx = common['task2idx']
    data = pickle.load(open(args['data_dir']+'/'+args['data_file'], 'rb'))
    test_data = data['test_data']
    test_data = [r['feats'] for r in test_data]

    emb = [None, None]
    if args['word_rnn']:
        word = pickle.load(open(args['data_dir']+'/'+args['word_emb'], 'rb'))
        word2idx = word['word2idx']
        word_emb = word['word_emb']
        for data_ in [test_data]:
            for d in data_:
                d['words'] = [word2idx[r.lower()] if r.lower() in word2idx else word2idx['<unk>'] for r in d['words']]
        emb[1] = word_emb
        args['pad_word_idx'] = word2idx['<pad>']

    if args['char_rnn']:
        char = pickle.load(open(args['data_dir']+'/'+args['char_emb'], 'rb'))
        char2idx = char['char2idx']
        char_emb = char['char_emb']
        emb[0] = char_emb
        args['pad_char_idx'] = char2idx['<pad>']


    test_data = [r for r in test_data if r['task']==task2idx[args['target_task']]]
    test_size = len(test_data)
    test_data = pack_data(args, test_data)
    model_ = CN_2(args, emb)
    model_.cuda()
    model_.load_state_dict(saved['state_dict'])
    model_.eval()
    task_attn_all = np.zeros((len(task2idx)))
    for batch in test_data:
        lstm_out, task_attn = super(CN_2, model_).forward(batch['data_input'], batch['sent_end'])
        task_attn_all += task_attn.sum(0).data.cpu().numpy()

    task_attn_all /= test_size
    result = sorted(list(zip([r[0] for r in sorted(list(task2idx.items()), key=lambda x:x[1])], list(task_attn_all))), key=lambda x:abs(x[1]), reverse=True)
    print('Target task:', args['target_task'])
    for task, score in result:
        if task != args['target_task']:
            print(task, '%.4f' % score)
    
    print()