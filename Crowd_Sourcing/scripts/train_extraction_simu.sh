export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

SEL='latent'
bs=32

DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset/"
for ANUM in 3 #5 10 15 30
do
    dir=num_$ANUM
    CORPUS=$DATA_PATH$dir/ner_dataset_ma.pk
    CP_ROOT=$EXP_PATH$dir
    for TASK in 'maAddVecCrowd+latent' #'maMulScoreCrowd+latent' 'maCatVecCrowd+latent'
    do
        CPNAME=Teachers_$TASK
        for rs in 999 #73 2 
        do
            python train_seq_extraction.py --corpus $CORPUS --task $TASK --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --selecting $SEL 
        done
    done
done
            
