#!/bin/bash
if [ ! -d "$1" ]; then
  echo "$1 direcotry does not exist"
else
  rm -r $1/output_model
  rm -r $1/tokenizer
  rm $1/oof_df.pkl
fi
