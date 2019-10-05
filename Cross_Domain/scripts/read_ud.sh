#!/bin/bash

source activate cn_ner

# 1. data_type, 2. load_data_dir, 3. save_data_dir
SRC_FOLDER="/.."
LOG_FOLDER="$SRC_FOLDER/logs"


cd $SRC_FOLDER
mkdir -p $LOG_FOLDER

python -u $SRC_FOLDER/read_ud.py \
  -data_dir $SRC_FOLDER/data/ud-treebanks-v2.3/UD_English-GUM \
  -word_emb_dir $SRC_FOLDER/word_embeddings \
  -save_data $SRC_FOLDER/data/ud-treebanks-v2.3_processed/UD_English-GUM \
  | tee $LOG_FOLDER/read_ud.log

conda deactivate