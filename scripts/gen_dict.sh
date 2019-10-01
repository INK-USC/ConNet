export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
DEV_DATA="/data/ouyu/data/conll2003/ner/raw/valid"
TEST_DATA="/data/ouyu/data/conll2003/ner/raw/test"

EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"


green=`tput setaf 2`
reset=`tput sgr0`

# pre-processing
echo ${green}=== Generating Dictionary ===${reset}

TRAIN_DATA="/data/ouyu/data/conll2003/ner/raw/train"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/raw/conll_map.pk"
#python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA --output_map $OUTPUT_DATA

#DATASET='comb6'
#TRAIN_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/"$DATASET"/train"
#OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/"$DATASET"/conll_map.pk"
#python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA --output_map $OUTPUT_DATA
#echo save conll_map.pk to $DATASET

#DATASET='comb_anno0.6_miss0.1_3'
#TRAIN_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/"$DATASET"/train"
#OUTPUT_DATA="/data/ouyu/data/conll2003/ner/multi-simulate/comb/conll_map.pk"
#python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA --output_map $OUTPUT_DATA

#TRAIN_DATA="/data/ouyu/data/conll2003/ner/raw/train"
#OUTPUT_DATA="/data/ouyu/data/conll2003/ner/raw/conll_map.pk"
#python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA --output_map $OUTPUT_DATA

#TRAIN_DATA='/data/ouyu/data/conll2003/ner/ner-mturk/train_answers_mv.txt'
#OUTPUT_DATA='/data/ouyu/data/conll2003/ner/ner-mturk/train_answers_mv.pk'
#python pre_seq/gene_map_labels.py --train_corpus $TRAIN_DATA --output_map $OUTPUT_DATA

#TRAIN_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/pruned_ground_truth.txt"
#OUTPUT_DATA="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/conll_map.pk"
TRAIN_DATA="/data/ouyu/data/conll2003/ner/raw_bio/train"
OUTPUT_DATA="/data/ouyu/data/conll2003/ner/raw_bio/conll_map.pk"
#python pre_seq/gene_map_labels.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA --output_map $OUTPUT_DATA
#python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA --output_map $OUTPUT_DATA

TRAIN_DATA='/home/ron_data/ouyu/data/panx_dataset/clean_data/train'
OUTPUT_DATA='/home/ron_data/ouyu/data/panx_dataset/clean_data/train_map.pk'
python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --output_map $OUTPUT_DATA

