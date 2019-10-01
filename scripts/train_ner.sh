export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
green=`tput setaf 2`
reset=`tput sgr0`

# train & test
echo ${green}=== Training ===${reset}

DATASET='ner'
SEQMODEL='vanilla'
for TASK in 'true' #'voteTOK' 'voteSEQ'
do
  #CORPUS="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/ner_dataset.pk"
  CORPUS=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_73_randseed_999/train_allset.pk
  DATASET=pruned_mturk_Teachers_maMulScoreCrowd_train_allset
  CPNAME=$DATASET"/NER"
  for rs in 999 73 2    
  do
    python train_seq.py --corpus $CORPUS --checkpoint_name $CPNAME --seq_model $SEQMODEL --rand_seed $rs
  done
done
