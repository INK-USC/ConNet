import sys
import numpy as np
import sys
sys.path.insert(0, '.')
import conlleval
from prettytable import PrettyTable


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

def read_file(f):
    snt = []
    snts = []
    answers = []
    for line in f:
        line = line.strip()
        if line == '':
            snts.append(' '.join(snt[0]))
            answers.append(snt[1:])
            snt = []
            continue
        if snt == []:
            snt = [[word] for word in line.split()]
        else:
            for idx, word in enumerate(line.split()):
                snt[idx].append(word)
    return snts, answers

def data_to_output(sentences, write_to_file=''):
  """Convert data to a string of data stream that is ready to write

  Args:
      sentences (list): A list of sentences
      write_to_file (str, optional):
        If a file path, write data stream to that file

  Returns:
      output_list: A list of strings, each line indicating a line in file
  """
  output_list = []
  for sentence in sentences:
    for tup in sentence:
      output_list.append('\t'.join(tup))
    output_list.append('')
  if write_to_file:
    with open(write_to_file, 'w') as f:
      f.write('\n'.join(output_list))
  return output_list

def extract_columns(sentences, indexs):
  """Extract columns of information from sentences
  
  Args:
      sentences (list): 
        Sentences is a list of sentences.
        Sentence is a list of token information.
        Token information is in format: [token, feature_1, ..., feature_n, tag_label] 

      indexs (list): A list of indexs to retrieve from. Can be positive
        or negative (backward)
  
  Returns:
      columns (list): Same format as sentences.
  """
  columns = []
  for sentence in sentences:
    columns.append([[tup[i] for i in indexs] for tup in sentence])
  return columns 

def extend_columns(sentences, columns):
  """Extend column to list of sentences
  
  Args:
      sentences (list): 
        Sentences is a list of sentences.
        Sentence is a list of token information.
        Token information is in format: [token, feature_1, ..., feature_n, tag_label] 
      columns (list): Same format as sentence
  
  Returns:
      new_sentences: same format as sentences
  """
  new_sentences = []
  for sentence, column in zip(sentences, columns):
    new_sentences.append(
        [tup + col for tup, col in zip(sentence, column)]
    )
  return new_sentences


#truth_file="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/pruned_ground_truth.txt"
#truth_file="/data/ouyu/data/conll2003/ner/raw_bio/" + sys.argv[1]
#truth_file="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/test_nodoc" #pruned_ground_truth.txt"
truth_file="/data/ouyu/data/conll2003/ner/ner-mturk/testset_pruned.txt"

"""
mv_file='train_mv'
truth_data = read_data(truth_file)
mv_data = read_data(mv_file)
p, r, f = conlleval.evaluate(data_to_output(mv_data))
"""

mv_file=sys.argv[1]
#mv_file = 'pruned_mv.txt'
truth_data = read_data(truth_file)
mv_data = read_data(mv_file)
mv_snt = extract_columns(mv_data, [-1])
new_snt = extend_columns(truth_data, mv_snt)
p, r, f = conlleval.evaluate(data_to_output(new_snt))

print(f)
