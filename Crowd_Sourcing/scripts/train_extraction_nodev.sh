export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
DATASET="comb3"

EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

#CORPUS="/data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/ner_dataset_ma.pk"
#DATASET='pruned_mturk'
CORPUS="/data/ouyu/data/conll2003/ner/multi-simulate/mturk/ner_dataset_ma.pk"
DATASET='mturk'
ANUM=47
#DATASET='comb_anno0.6_miss0.1_3'
#ANUM=3

TASK='maMulVecCrowd+latent' #"maMulCRFCrowd+latent"
CPNAME=$DATASET"/Teachers_"$TASK
#SEL='avg' 
SEL='latent'
bs=32
for TASK in 'maAddVecCrowd+latent' 'maCatVecCrowd+latent' 'maMulScoreCrowd+latent' 'maMulMatCrowd+maMulCRFCrowd+latent'
do
CPNAME=$DATASET"/Teachers_"$TASK
for rs in 999 73 2
do
  python train_seq_extraction_nodev.py --corpus $CORPUS --task $TASK --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --selecting $SEL 
done
done
