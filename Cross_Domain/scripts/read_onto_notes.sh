#!/bin/bash

source activate cn_ner

SRC_FOLDER="/.."
LOG_FOLDER="$SRC_FOLDER/logs"


cd $SRC_FOLDER
mkdir -p $LOG_FOLDER

python -u $SRC_FOLDER/read_onto_notes.py \
  -data_dir $SRC_FOLDER/data/OntoNotes-5.0-NER-BIO-OntoNER/conll-formatted-ontonotes-5.0/data \
  -word_emb_dir $SRC_FOLDER/word_embeddings \
  -save_data $SRC_FOLDER/data/OntoNotes-5.0-NER-BIO-OntoNER_processed \
  | tee $LOG_FOLDER/read_onto_notes.log

conda deactivate