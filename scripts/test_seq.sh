export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

bs=32
NOISE_MOD=none #crf_transition #all
MODEL_NUM=50
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_ground_truth_subsets/"$MODEL_NUM
TASK='true'
SEQMODEL='crf'

for file in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
do
for rs in  999 #11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
  #CORPUS=/home/ron_data/ouyu/data/conll2003/ner/raw_bio/ner_dataset.pk
  #DATASET=raw_bio
  CORPUS_TEST=/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_true.pk
  CORPUS=/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_ground_truth_subsets/$MODEL_NUM/ner_dataset_${file}_true.pk
  DATASET=pruned_mturk_split_${MODEL_NUM}_${file}
  CPNAME=$DATASET"/NER"
  PREMODEL=/home/ron_data/ouyu/exp/CN_NER/checkpoint/${DATASET}/NER/crf_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_epoch_100_randseed_999
  python test_seq.py --corpus $CORPUS --corpus_test $CORPUS_TEST --checkpoint_name $CPNAME --rand_seed $rs --batch_size $bs --restore_model_path $PREMODEL --noise_module $NOISE_MOD --seq_model $SEQMODEL 
done
done
