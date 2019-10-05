export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
DATASET="comb3"

EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

#CORPUS="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_ma.pk"
#DATASET='pruned_mturk'
#CORPUS="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_ma.pk"
#DATASET='mturk'
#ANUM=47
#CPNAME=$DATASET"/Teachers_"$TASK

DATASET_TYPE="simulate_dataset_model"
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/"$DATASET_TYPE/
EXP_PATH="/home/ron_data/ouyu/exp/"$DATASET_TYPE/
TASK="maMulScoreCrowd" #'maAddVecCrowd+latent'
DATASET=subset_10 #"0.1x50_3"
ANUM=10

SEL='avg' 
#SEL='latent'
bs=32

for DATASET in subset_50_50 #subset_50_10_10 subset_50_15_15 subset_50_30_30 #subset_5_5 subset_10_5_5 subset_15_5_5 subset_30_5_5 subset_50_5_5 #subset_50_10_10 subset_50_15_15 subset_50_30_30 subset_50_50
do
  echo == $DATASET ==
  for rs in 999 73 2
  do
  CORPUS=$DATA_PATH$DATASET/ner_dataset_ma.pk
  CP_ROOT=$EXP_PATH$DATASET
  CPNAME=Teachers_$TASK
  PREMODEL=$CP_ROOT/$CPNAME+latent/randCrowd_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_${rs}
  python test_seq_extraction.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --selecting $SEL --restore_model_path $PREMODEL 
  done
done
