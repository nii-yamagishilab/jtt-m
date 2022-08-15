#!/bin/bash

set -ex

base_dir="$(basename "$PWD")"

# Remove suffix until the "+" character
data_name="${base_dir%+*}"
data_dir="../data/${data_name}"

pretrained='bert-base-uncased'
max_len=128
out_dir="${pretrained}-${max_len}-out"

python '../../mahalanobis_scorer.py' \
  --gold_file "${data_dir}"/train.jsonl \
  --emb_file "${out_dir}"/train.emb.npy
