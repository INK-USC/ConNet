export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
DATASET="comb3"

EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

DATA_PATH="/home/ron_data/ouyu/data/conll2003/ner/simulate_dataset/"
EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset/"
PARAMS="randCrowd_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_"

SEL='avg' 
#SEL='latent'
bs=32

for ANUM in 3 #5 10 15 30
do
    dir=num_$ANUM
    CORPUS=$DATA_PATH$dir/ner_dataset_ma.pk
    CP_ROOT=$EXP_PATH$dir
    for TASK in 'maMulScoreCrowd+latent' 
    do
        CPNAME=Teachers_$TASK
        for rs in 999 #73 2 
        do
            PREMODEL=$CP_ROOT/$CPNAME/$PARAMS$rs
            python test_seq_extraction.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --selecting $SEL --restore_model_path $PREMODEL 
        done
    done
done
