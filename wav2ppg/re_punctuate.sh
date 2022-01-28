#!/usr/bin/env bash


. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <text-path> <punctuated_path>"
  echo " $0 data/matbn/train_hires/text data/matbn/train_hires/punctuated_text"
  exit 1;
fi
echo "$0 $@"

text=$1
punctuated=$2

python /home/yuhuei/yuhuei/punctuation/punctuate_inference.py \
    $text $punctuated