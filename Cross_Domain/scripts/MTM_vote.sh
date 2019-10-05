#!/bin/bash

source activate cn_ner

EXEC_NAME=MTM_vote_$1

SRC_FOLDER="/.."
LOG_FOLDER="$SRC_FOLDER/logs/MTM_vote"


cd $SRC_FOLDER
mkdir -p $LOG_FOLDER

python -u $SRC_FOLDER/MTM_vote.py \
  -checkpoint $SRC_FOLDER/checkpoints/$2 \
  | tee $LOG_FOLDER/$EXEC_NAME.log

conda deactivate