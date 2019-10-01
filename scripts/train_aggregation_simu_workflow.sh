export CUDA_DEVICE_ORDER=PCI_BUS_ID

# modify the following section
EMBEDDING_DATA="/home/ron_data/ouyu/data/embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`


# train & test
echo ${green}=== Training ===${reset}

MAP_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/conll_map.pk"
DEV_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/valid"
TEST_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/test"

se='hdtl'
me=5
bs=32
SEL='avg'

for MODELUP in 'False' #'True'
do
  rs=2
  TMF1="0.77846093 0.72369328 0.80609609 0.74088478 0.67390202 0.74931708 0.76897806 0.84507042 0.73960087 0.79872881 0.83955864 0.81480255 0.83036351 0.81724846 0.82582728 0.74027085 0.82564432 0.70219436 0.69471716 0.78408979 0.8403109  0.81318338 0.81525749 0.81024306 0.7146283  0.81836854 0.74697499 0.80668617 0.78603089 0.73454257 0.73460722 0.69296375 0.7863021  0.80202004 0.7604649  0.82882542 0.8411547  0.80722326 0.8242441  0.71323823 0.83147291 0.62932129 0.82233991 0.73385301 0.76743098 0.81444099 0.70725631 0.73427762 0.78379634 0.83845959"
  #TASK="maAddVecCrowd+latent"
  TASK="maMulScoreCrowd" #+latent"
  DATASET=subset_50_50 #subset_10_3 #"0.1x50_3" #type_normal__acc_0.9__sigma_0.1__num_50_3
  ANUM=50
  EXP_PATH="/home/ron_data/ouyu/exp/simulate_dataset_model"
  TEACHER_EXP_PATH=${EXP_PATH}/${DATASET}/Teachers_${TASK}
  TRAIN_PATH=randCrowd_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_${rs}
  TEST_PATH=test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_${SEL}_epoch_200_batchSize_32_randseed_${rs}
  TRAIN_DATA=$TEACHER_EXP_PATH/$TEST_PATH/train_allset
  CORPUS=$TEACHER_EXP_PATH/$TEST_PATH/train_allset.pk
  TM=$TEACHER_EXP_PATH+latent/$TRAIN_PATH
  CPNAME=Student_$TASK
  CP_ROOT=${EXP_PATH}/$DATASET

  python scripts/weighted_vote.py --TF "$TMF1" --datapath $TRAIN_DATA

  python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $CORPUS --task true 
  
  python train_seq_aggregation.py --corpus $CORPUS --task $TASK --cp_root $CP_ROOT --checkpoint_name $CPNAME --annotator_num $ANUM --rand_seed $rs --batch_size $bs --restore_teachers_path $TM --init_teachers_reliability "${TMF1}" --model_require_update $MODELUP --snt_emb $se --epoch $me  
done
