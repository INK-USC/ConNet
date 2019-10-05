export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}


se='hdtl'
me=5
bs=32
for MODELUP in 'False' #'True'
do
for rs in 999 #73 2
do
  TASK="maMulScoreCrowd"
  DATASET=pruned_mturk
  ANUM=47
  EXP_PATH="/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint"
  CORPUS=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999/train_allset_bea.pk
  TM=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/randCrowd_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999
  CPNAME=Student_$TASK
  CP_ROOT=${EXP_PATH}/$DATASET
  TMF1="0.15699146 0.26383636 0.10696469 0.13220468 0.07346614 0.13873539 0.08013783 0.23486405 0.0830624  0.2524096  0.12751929 0.12414168 0.06288738 0.13376789 0.26940979"
  
  python train_seq_aggregation.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --restore_teachers_path $TM --init_teachers_reliability "${TMF1}" --model_require_update $MODELUP --snt_emb $se --epoch $me  
done
done
