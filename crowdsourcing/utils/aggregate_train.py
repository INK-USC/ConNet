import sys

def read_data(fname, ignore_docstart=False):
  """Read data from any files with fixed format.
  Each line of file should be a space-separated token information,
  in which information starts from the token itself.
  Each sentence is separated by a empty line.

  e.g. 'Apple NP (NP I-ORG' could be one line

  Args:
      fname (str): file path for reading data.

  Returns:
      sentences (list):
        Sentences is a list of sentences.
        Sentence is a list of token information.
        Token information is in format: [token, feature_1, ..., feature_n, tag_label]   
  """
  sentences, prev_sentence = [], []
  with open(fname) as f:
    for line in f:
      if not line.strip():
        if prev_sentence and (not ignore_docstart or len(prev_sentence) > 1):
          sentences.append(prev_sentence)
        prev_sentence = []
        continue
      prev_sentence.append(list(line.strip().split()))
  if prev_sentence != []:
      sentences.append(prev_sentence)
  return sentences

def read_data_to_list(f):
    data = read_data(f)
    word_list = []
    tag_list = []
    for snt in data:
        words = ' '.join([tok[0].lower() for tok in snt])
        tags = ' '.join([tok[-1] for tok in snt])
        word_list.append(words)
        tag_list.append(tags)
    return word_list, tag_list
    

name = sys.argv[1]
truth_file="/data/ouyu/data/conll2003/ner/ner-mturk/ground_truth.txt"
#truth_file="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/pruned_ground_truth.txt"
files = [name+'_'+str(i) for i in range(47)]
truth_data = open(truth_file, 'r').readlines()
datasets = [open(f, 'r').readlines() for f in files]

out_file=open(name, 'w')
for idx, line in enumerate(truth_data):
    snt = []
    if line.strip() == '':
        out_file.write('\n')
        continue
    w = line.strip().split()[0]
    snt.append(w)
    for i in range(47):
        t = datasets[i][idx].strip().split()[-1]
        snt.append(t)
    out_file.write(' '.join(snt)+'\n')



