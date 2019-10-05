#!/bin/bash

source activate cn_ner

SRC_FOLDER="/.."
LOG_FOLDER="$SRC_FOLDER/logs"


cd $SRC_FOLDER
mkdir -p $LOG_FOLDER

python -u $SRC_FOLDER/read_cross_lingual_emb.py \
  | tee $LOG_FOLDER/read_cross_lingual_emb.log

conda deactivate