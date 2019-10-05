export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`

# train & test
echo ${green}=== Testing ===${reset}

TASK="maMulScoreCrowd"
CPROOT="/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint"
CPNAME="EXP_TEACHERS_MULSCORE_VOTEWTOK/Student_maMulScoreCrowd+latent"
CORPUS="/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999/train_mturk_allset/ner_dataset_voteWTOK.pk"
PREMODEL="/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/EXP_TEACHERS_MULSCORE_VOTEWTOK/Student_maMulScoreCrowd+latent/matrix_hdtl_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_batchSize_32_modelUpdate_True_randseed_999_batchSize_32"
bs=1
rs=1
anum=47
se='hdtl'

python test_seq_aggregation.py --corpus $CORPUS --cp_root $CPROOT --checkpoint_name $CPNAME --rand_seed $rs --batch_size $bs --restore_model_path $PREMODEL --task $TASK --annotator_num $anum --snt_emb $se 
