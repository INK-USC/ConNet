export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
green=`tput setaf 2`
reset=`tput sgr0`

# train & test
echo ${green}=== Training ===${reset}

DATASET='ner'
SEQMODEL='crf'
TASK='true'
NUM_FILE=30
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_ground_truth_subsets/"${NUM_FILE}
for file in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
do
  CORPUS=$DATA_PATH/ner_dataset_${file}_$TASK.pk
  DATASET=pruned_mturk_split_${NUM_FILE}_${file}
  CPNAME=$DATASET"/NER"
  for rs in 999 #73 2    
  do
    python train_seq.py --corpus $CORPUS --checkpoint_name $CPNAME --seq_model $SEQMODEL --rand_seed $rs
  done
done
