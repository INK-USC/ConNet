export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

bs=32
NOISE_MOD=crf_transition #all

for SIGMA in 0.4 0.5 0.6 0.7
do
for rs in  1 #11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
  #CORPUS=/home/ron_data/ouyu/data/conll2003/ner/raw_bio/ner_dataset.pk
  #DATASET=raw_bio
  CORPUS=/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_true.pk
  DATASET=pruned_mturk_true
  PREMODEL=/home/ron_data/ouyu/exp/CN_NER/checkpoint/pruned_mturk_true/NER/origin_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_epoch_200_randseed_888
  CPNAME=$DATASET"/NER"
  python test_seq.py --corpus $CORPUS  --checkpoint_name $CPNAME --rand_seed $rs --batch_size $bs --restore_model_path $PREMODEL --sigma $SIGMA --noise_module $NOISE_MOD 
done
done
