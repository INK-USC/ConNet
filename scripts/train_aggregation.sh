export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

TASK="maMulScoreCrowd+latent"

se='hdtl'
me=5
bs=32
for MODELUP in 'False' #'True'
do
for rs in 999 73 2
do
  DATASET='mturk'
  CPNAME=$DATASET"/Student_"$TASK
  DATAPATH=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_${rs}
  TM=/data/ouyu/exp/Vanilla_NER/checkpoint/mturk/Teachers_maMulScoreCrowd+latent/nodev_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_50_batchSize_32_randseed_${rs}
  
  CORPUS="/home/ron_data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_voteTOK.pk"
  TMF1="0.41520632 0.0434116  0.37364681"
  python train_seq_aggregation_nodev.py --corpus $CORPUS --task $TASK --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --restore_teachers_path $TM --init_teachers_reliability "${TMF1}" --model_require_update $MODELUP --snt_emb $se --epoch $me 
done
done
