export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`

# train & test
echo ${green}=== Training ===${reset}

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset/"
for dir in num_3 num_5 num_10 num_15 num_30
do
for rs in 999 73 2
do
    CORPUS=$DATA_PATH$dir/ner_dataset_voteTOK.pk
    CP_ROOT=$EXP_PATH$dir
    CPNAME=vote_tok
    python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs

    CORPUS=$DATA_PATH$dir/ner_dataset_voteSEQ.pk
    CP_ROOT=$EXP_PATH$dir
    CPNAME=vote_seq
    python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs
done
done
fi


execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_type/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_type/"
  for dir in `ls $DATA_PATH`
  do
    echo == $dir ==
    for rs in 999 73 2
    do
      for TASK in 'vote_tok' 'vote_seq'
      do
        CORPUS=$DATA_PATH$dir/ner_dataset_$TASK.pk
        CP_ROOT=$EXP_PATH$dir
        CPNAME=$TASK
        python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs
      done
    done
  done
fi


execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_type_truc/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_type_truc/"
  for dir in type_normal__acc_0.9__sigma_0.1__num_50_3 #`ls $DATA_PATH`
  do
    echo == $dir ==
    for rs in 999 73 2
    do
      for TASK in 'vote_tok' 'vote_seq'
      do
        CORPUS=$DATA_PATH$dir/ner_dataset_$TASK.pk
        CP_ROOT=$EXP_PATH$dir
        CPNAME=$TASK
        python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs
      done
    done
  done
fi

execute=true
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_model/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_model/"
  for dir in subset_5_5 subset_10_5_5 subset_15_5_5 subset_30_5_5 subset_50_5_5 subset_50_10_10 subset_50_15_15 subset_50_30_30 subset_50_50
  do
    echo == $dir ==
    for rs in 999 73 2
    do
      for TASK in 'vote_tok' #'vote_seq'
      do
        CORPUS=$DATA_PATH$dir/ner_dataset_$TASK.pk
        CP_ROOT=$EXP_PATH$dir
        CPNAME=$TASK
        python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs
      done
    done
  done
fi
