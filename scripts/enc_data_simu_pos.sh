export CUDA_DEVICE_ORDER=PCI_BUS_ID

green=`tput setaf 2`
reset=`tput sgr0`

# pre-processing
echo ${green}=== Encoding ===${reset}

MAP_DATA="/home/ron_data/ouyu/data/wsj/raw/map.pk"
DEV_DATA="/home/ron_data/ouyu/data/wsj/raw/valid"
TEST_DATA="/home/ron_data/ouyu/data/wsj/raw/test"

DATA_PATH="/home/ron_data/ouyu/data/wsj/simulate_dataset/"
for dir in num_3 num_5 num_10 num_15 num_30
do
    echo == $dir ==
    TRAIN_DATA=$DATA_PATH$dir/train
    OUTPUT_DATA=$DATA_PATH$dir/pos_dataset_ma.pk
    TASK='ma'
    python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

    OUTPUT_DATA=$DATA_PATH$dir/pos_dataset_voteTOK.pk
    TASK='vote_tok'
    python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

    OUTPUT_DATA=$DATA_PATH$dir/pos_dataset_voteSEQ.pk
    TASK='vote_seq'
    python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
done
