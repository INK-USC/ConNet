export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`

# train & test
echo ${green}=== Training ===${reset}

DATA_PATH="/home/ron_data/ouyu/data/wsj/simulate_dataset/"
EXP_PATH="/home/ron_data/ouyu/exp/wsj/simulate_dataset/"
for dir in num_3 #num_5 num_10 num_15 num_30
do
for rs in 999 #73 2
do
    CORPUS=$DATA_PATH$dir/pos_dataset_voteTOK.pk
    CP_ROOT=$EXP_PATH$dir
    CPNAME=vote_tok
    python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs

    CORPUS=$DATA_PATH$dir/pos_dataset_voteSEQ.pk
    CP_ROOT=$EXP_PATH$dir
    CPNAME=vote_seq
#   python train_seq.py --corpus $CORPUS --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs
done
done
