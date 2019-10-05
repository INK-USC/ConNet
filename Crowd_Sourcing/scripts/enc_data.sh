export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
TRAIN_DATA="/data/ouyu/data/conll2003/ner/raw/train"
DEV_DATA="/data/ouyu/data/conll2003/ner/raw/valid"
TEST_DATA="/data/ouyu/data/conll2003/ner/raw/test"

EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

MAP_DATA="/data/ouyu/data/conll2003/ner/raw/conll_map.pk"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/raw/ner_dataset.pk"

TASK='true'

green=`tput setaf 2`
reset=`tput sgr0`

# pre-processing
echo ${green}=== Encoding ===${reset}
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

MAP_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/conll_map.pk"
DEV_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/valid"
TEST_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/test"

TRAIN_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/train"
OUTPUT_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/ner_dataset.pk"
python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA

#mturk
TRAIN_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/answers_unk.txt"
TEST_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/testset.txt"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_ma.pk"
TASK='ma'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_voteTOK.pk"
TASK='vote_tok'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_voteSEQ.pk"
TASK='vote_seq'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

TRAIN_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/ground_truth.txt"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_true.pk"
TASK='true'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

#mturk-devtest
TRAIN_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/answers_unk.txt"
DEV_DATA="/data/ouyu/data/conll2003/ner/raw_bio/valid"
TEST_DATA="/data/ouyu/data/conll2003/ner/raw_bio/test"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk_devtest/ner_dataset_ma.pk"
TASK='ma'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_voteTOK.pk"
TASK='vote_tok'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_voteSEQ.pk"
TASK='vote_seq'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK


TRAIN_DATA='/data/ouyu/data/conll2003/ner/ner-mturk/train_answers_mv.txt'
MAP_DATA='/data/ouyu/data/conll2003/ner/ner-mturk/train_answers_mv.pk'
OUTPUT_DATA='/data/ouyu/data/conll2003/ner/ner-mturk/ner_dataset_aggregator.pk'
TASK='ma'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

# pruned mturk
#TRAIN_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/train"
DEV_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/valid"
TEST_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/test"
MAP_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/conll_map.pk"
#OUTPUT_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_ma.pk"
DATAPATH="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk"
TRAIN_DATA=${DATAPATH}/train
OUTPUT_DATA=${DATAPATH}/ner_dataset_ma.pk
TASK='ma'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

OUTPUT_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_voteTOK.pk"
TASK='vote_tok'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

OUTPUT_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_voteSEQ.pk"
TASK='vote_seq'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

TRAIN_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/pruned_ground_truth.txt"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_true.pk"
Task='true'
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK

# pruned mturk answers
DEV_DATA="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/valid"
TEST_DATA="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/test"
MAP_DATA="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/conll_map.pk"
TRAIN_DATAPATH="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/answers"
OUTPUT_DATA=${TRAIN_DATAPATH}/answers.pk
ANUM=47
#python pre_seq/encode_data_multitrain.py --train_path $TRAIN_DATAPATH --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --a_num $ANUM

# panx dataset
DATAPATH='/home/ron_data/ouyu/data/panx_dataset/clean_data'
MAP_DATA=$DATAPATH/train_map.pk
if false; then
for fold in $DATAPATH/*
do
    if [ -d $fold ]; then
        echo $fold
        TRAIN_DATA=$fold/train
        DEV_DATA=$fold/dev
        TEST_DATA=$fold/test
        OUTPUT_DATA=$fold/dataset.pk
        #python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA 
    fi
done
fi

TRAIN_DATA=$DATAPATH/train
DEV_DATA=$DATAPATH/dev
TEST_DATA=$DATAPATH/test
OUTPUT_DATA=$DATAPATH/dataset.pk
#python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA 

