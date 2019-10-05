export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

DATAPATH='/home/ron_data/ouyu/data/panx_dataset/clean_data'
CORPUS=`ls $DATAPATH/*/*.pk`
CORPUS=$CORPUS' '$DATAPATH/dataset.pk
CP_ROOT='/home/ron_data/ouyu/exp/CN_MULTILINGUAL/checkpoint'
DATASET='PANX'
ANUM=41
BS=32
LA=0.8

for TASK in 'maMulScoreCrowd+latent' 
do
  CPNAME=$DATASET"/CN_"$TASK
  SEL='latent'
  for rs in 999 #888 777
  do
    echo ${green}=== rs is $rs ===${reset}
    LR=0.1
    python train_mling_cn.py --corpus "${CORPUS}" --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $BS --selecting $SEL --lr $LR 
    #python train_mling_cn.py --corpus "${CORPUS}" --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $BS --selecting $SEL --loss_att --lambda_att $LA 
    #python train_mling_cn.py --corpus "${CORPUS}" --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $BS --selecting $SEL --loss_att --lambda_att $LA --test_att 
    #python train_mling_cn.py --corpus "${CORPUS}" --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $BS --selecting $SEL --loss_att --lambda_att $LA --test_att --train_att
  done
done
