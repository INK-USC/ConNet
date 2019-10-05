#!/bin/bash

source activate cn_ner

EXEC_NAME=GUM_${8}_${1}

SRC_FOLDER="/.."
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/UD-GUM/$EXEC_NAME"
LOG_FOLDER="$SRC_FOLDER/logs/UD-GUM"


cd $SRC_FOLDER
mkdir -p $CHECKPOINT_FOLDER
mkdir -p $LOG_FOLDER

python -u $SRC_FOLDER/train.py \
  -data_dir $SRC_FOLDER/data/ud-treebanks-v2.3_processed/UD_English-GUM \
  -data_file data.p \
  -evaluator token \
  -char_rnn \
  -char_emb random_char_emb_${2}.p \
  -char_rnn_hid ${3} \
  -word_rnn \
  -word_emb glove.6B.${4}d.txt.p \
  -word_rnn_hid ${5} \
  -batch 32 \
  -epochs 200 \
  -patience 10 \
  -seed ${6} \
  -lr ${7} \
  -dropout 0.5 \
  -cuda \
  -save_model $CHECKPOINT_FOLDER \
  -model ${8} \
  -cm_dim ${9} \
  -fine_tune ${10} \
  -mode ${11} \
  -target_task ${12} \
  -down_sample 50 \
  | tee $LOG_FOLDER/$EXEC_NAME.log

conda deactivate