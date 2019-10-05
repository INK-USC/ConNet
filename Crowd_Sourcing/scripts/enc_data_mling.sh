export CUDA_DEVICE_ORDER=PCI_BUS_ID
green=`tput setaf 2`
reset=`tput sgr0`

echo ${green}=== Encoding ===${reset}

TRAIN_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data"
MAP_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/map.pk"
OUTPUT_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/dataset.pk"
LOG_FILE="/home/molly_data/ouyu/data/panx_dataset/raw_data/log.dataset"
LANGS="af ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it lt lv mk ms nl no pl pt ro ru sk sl sq sv ta tl tr uk vi"

#python pre_seq/encode_data_mling.py --train_file_path $TRAIN_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --log_file $LOG_FILE --langs "$LANGS" 

for lang in af ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it lt lv mk ms nl no pl pt ro ru sk sl sq sv ta tl tr uk vi
do
echo $lang
TRAIN_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/$lang/train"
TEST_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/$lang/test"
DEV_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/$lang/dev"
MAP_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/$lang/map.pk"
OUTPUT_DATA="/home/molly_data/ouyu/data/panx_dataset/raw_data/$lang/dataset.pk"

python pre_seq/encode_data.py --train_file $TRAIN_DATA --dev_file $DEV_DATA --test_file $TEST_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA  

done
