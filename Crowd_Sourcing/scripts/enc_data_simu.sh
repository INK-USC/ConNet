export CUDA_DEVICE_ORDER=PCI_BUS_ID

green=`tput setaf 2`
reset=`tput sgr0`

# pre-processing
echo ${green}=== Encoding ===${reset}

MAP_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/conll_map.pk"
DEV_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/valid"
TEST_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/test"

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset/"
  for dir in num_3 num_5 num_10 num_15 num_30
  do
    echo == $dir ==
    TRAIN_DATA=$DATA_PATH$dir/train
    OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_ma.pk
    TASK='ma'
    python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

    OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_voteTOK.pk
    TASK='vote_tok'
    python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

    OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_voteSEQ.pk
    TASK='vote_seq'
    python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
  done
fi

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_type/"
  for dir in `ls $DATA_PATH`
  do
    echo == $dir ==
    TRAIN_DATA=$DATA_PATH$dir/train
    echo == $TRAIN_DATA ==
    for TASK in 'ma' 'vote_tok' 'vote_seq'
    do
        OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_$TASK.pk
        python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
    done
  done
fi

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_type_truc/"
  for dir in type_normal__acc_0.9__sigma_0.1__num_50_3  #`ls $DATA_PATH`
  do
    echo == $dir ==
    TRAIN_DATA=$DATA_PATH$dir/train
    echo == $TRAIN_DATA ==
    for TASK in 'ma' 'vote_tok' 'vote_seq'
    do
        OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_$TASK.pk
        python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
    done
  done
fi

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset_model/"
  for dir in subset_5_5 subset_10_5_5 subset_15_5_5 subset_30_5_5 subset_50_5_5 subset_50_10_10 subset_50_15_15 subset_50_30_30 subset_50_50
  do
    echo == $dir ==
    TRAIN_DATA=$DATA_PATH$dir/train
    echo == $TRAIN_DATA ==
    for TASK in 'ma' 'vote_tok' 'vote_seq'
    do
        OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_$TASK.pk
        python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
    done
  done
fi

execute=false
if [ $execute == true ]; then
DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_ground_truth_subsets/50"
  for file in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
  do
    echo == $file ==
    TRAIN_DATA=$DATA_PATH/train_$file
    echo == $TRAIN_DATA ==
    for TASK in 'true' #'ma' 'vote_tok' 'vote_seq'
    do
        OUTPUT_DATA=$DATA_PATH$dir/ner_dataset_${file}_$TASK.pk
        python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
    done
  done
fi

TASK=true
#TRAIN_DATA=/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/pruned_ground_truth.txt
#OUTPUT_DATA=/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_true.pk
TRAIN_DATA=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_73_randseed_999/train_allset
OUTPUT_DATA=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_73_randseed_999/train_allset.pk
python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
