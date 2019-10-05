export CUDA_DEVICE_ORDER=PCI_BUS_ID

#MAP_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/conll_map.pk"
MAP_DATA="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_mturk/conll_map.pk"
DEV_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/valid"
TEST_DATA="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/test"
TASK=true

TRAIN_DATA=/home/ron_data/ouyu/exp/simulate_dataset_type/type_normal__acc_0.4__sigma_0.2__num_15/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999/train_allset
OUTPUT_DATA=/home/ron_data/ouyu/exp/simulate_dataset_type/type_normal__acc_0.4__sigma_0.2__num_15/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999/train_allset.pk

TRAIN_DATA=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999/train_allset_bea
OUTPUT_DATA=/home/ron_data/ouyu/exp/Vanilla_NER/checkpoint/pruned_mturk/Teachers_maMulScoreCrowd+latent/test_unit_lstm_hidim_300_layer_1_drop_0.5_lr_0.015_lrDecay_0.05_opt_SGD_select_latent_epoch_200_batchSize_32_randseed_999/train_allset_bea.pk

python ~/workspace/Vanilla_NER/pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA --input_map $MAP_DATA --output_file $OUTPUT_DATA --task $TASK
