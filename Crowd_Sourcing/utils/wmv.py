"""
weighted majority voting
"""


import sys
from collections import Counter

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
    
mv_file = open(sys.argv[1]+'_wmv', 'w')
weighted_vector = sys.argv[2]
weighted_vector = [float(v) for v in weighted_vector.split()]
print('weighted_vector', weighted_vector)
with open(sys.argv[1], 'r') as f:
    for idx, line in enumerate(f):
        line = line.strip().split()
        if line == []:
            mv_file.write('\n')
        else:
            tok = [line[0]]
            #tags = [tag for tag in line[1:] if tag not in ["?", "<unk>"]]
            tags = line[1:]
            tag_dict = {tag : 0 for tag in tags}
            for aid, tag in enumerate(tags):
                tag_dict[tag] += weighted_vector[aid]

            sorted_tag_dict = sorted(tag_dict.items(), key=lambda kv: kv[1])

            if sorted_tag_dict[-1][0] in ["?", "<unk>"]:
                tok.append(sorted_tag_dict[-2][0])
            else:
                tok.append(sorted_tag_dict[-1][0])

            mv_file.write(' '.join(tok)+'\n')


