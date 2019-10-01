export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

SEL='latent'
bs=32

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset/"
for ANUM in 3 5 10 15 30
do
    dir=num_$ANUM
    CORPUS=$DATA_PATH$dir/ner_dataset_ma.pk
    CP_ROOT=$EXP_PATH$dir
    TASK='maMulScoreCrowd+latent' #'maMulScoreCrowd+latent' 'maCatVecCrowd+latent'
    CPNAME=Teachers_$TASK
    for rs in 999 73 2 
    do
        python train_seq_extraction.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --selecting $SEL 
    done
done
fi



execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_type/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_type/"
TASK='maMulScoreCrowd+latent'
  for dir in `ls $DATA_PATH`
  do
    echo == $dir ==
    for rs in 999 73 2
    do
        CORPUS=$DATA_PATH$dir/ner_dataset_ma.pk
        CP_ROOT=$EXP_PATH$dir
        CPNAME=Teachers_$TASK
        python train_seq_extraction.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs --batch_size $bs --selecting $SEL 
    done
  done
fi


execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_type_truc/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_type_truc/"
TASK='maMulScoreCrowd+latent'
for TASK in 'maMulScoreCrowd+latent' 'maAddVecCrowd+latent' 'maCatVecCrowd+latent'; do
  for dir in type_normal__acc_0.9__sigma_0.1__num_50_3 #`ls $DATA_PATH`
  do
    echo == $dir ==
    for rs in 999 73 2
    do
        CORPUS=$DATA_PATH$dir/ner_dataset_ma.pk
        CP_ROOT=$EXP_PATH$dir
        CPNAME=Teachers_$TASK
        python train_seq_extraction.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs --batch_size $bs --selecting $SEL 
    done
  done
done
fi

execute=true
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_model/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_model/"
TASK='maMulScoreCrowd+latent'
for TASK in 'maMulScoreCrowd+latent' #'maAddVecCrowd+latent' 'maCatVecCrowd+latent'
do
  for dir in subset_5_5 subset_10_5_5 subset_15_5_5 subset_30_5_5 subset_50_5_5 #subset_50_10_10 subset_50_15_15 subset_50_30_30 subset_50_50
  do
    echo == $dir ==
    for rs in 999 73 2
    do
        CORPUS=$DATA_PATH$dir/ner_dataset_ma.pk
        CP_ROOT=$EXP_PATH$dir
        CPNAME=Teachers_$TASK
        python train_seq_extraction.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs --batch_size $bs --selecting $SEL 
    done
  done
done
fi
