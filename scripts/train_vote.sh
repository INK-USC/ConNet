export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
DATASET="comb3"

EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

CORPUS_PATH="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_"
DATASET='pruned_mturk'
for TASK in 'vote_tok' #'vote_seq'
do
  CPNAME=$DATASET"/NER_"$TASK
  CORPUS=$CORPUS_PATH$TASK.pk
  for rs in 999 73 2
  do
   python train_seq.py --corpus $CORPUS --checkpoint_name $CPNAME --rand_seed $rs
  done
done
