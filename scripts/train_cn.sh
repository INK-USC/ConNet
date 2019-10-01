export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

CORPUS="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/answers/answers.pk"
DATASET='splited_pruned_mturk'
ANUM=47
BS=32
LA=0.8

for TASK in 'maMulScoreCrowd+latent' 
do
  CPNAME=$DATASET"/CN_"$TASK
  SEL='latent'
  for rs in 666 #999 888 777
  do
    python train_seq_cn.py --corpus $CORPUS --task $TASK --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $BS --selecting $SEL --lambda_att $LA 
  done
done
