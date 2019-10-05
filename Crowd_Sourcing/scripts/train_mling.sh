export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

MAP="/home/molly_data/ouyu/data/panx_dataset/raw_data/map.pk"
CORPUS="/home/molly_data/ouyu/data/panx_dataset/raw_data/dataset.pk"
CP_ROOT='/home/ron_data/ouyu/exp/CN_MULTILINGUAL/checkpoint'
DATASET='PANX'
TGT_LANG='ar'
BS=10
CPNAME=$DATASET"/NER_"$TGT_LANG

for rs in 999 #888 777
do
  echo ${green}=== rs is $rs ===${reset}
  GPU=0
  CUDA_VISIBLE_DEVICES=$GPU python train_mling.py --corpus ${CORPUS} --map $MAP --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs --batch_size $BS --tgt_lang $TGT_LANG --gpu $GPU
  #python train_mling.py --corpus ${CORPUS} --map $MAP --cp_root $CP_ROOT --checkpoint_name $CPNAME --rand_seed $rs --batch_size $BS --tgt_lang $TGT_LANG
done
