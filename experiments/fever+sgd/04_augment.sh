#!/bin/bash

set -ex

base_dir="$(basename "$PWD")"

# Remove suffix until the "+" character
data_name="${base_dir%+*}"
data_dir="../data/${data_name}"

# Replace '+' with '_'
aug_dir="${base_dir/+/_}"

pretrained='bert-base-uncased'
max_len=128
out_dir="${pretrained}-${max_len}-out"

# Upsampling rate
# Uncomment the following for another rate
#up=1
#up=2
up=3
#up=4

# With Mahalabobis
# Uncomment the following for another degree of freedom
#df=4
df=5
#df=6
python '../../augment.py' \
  --gold_file "${data_dir}/train.jsonl" \
  --prob_file "${out_dir}/train.prob.npy" \
  --mahal_file "${out_dir}/train.mahal.npy" \
  --aug_dir "${aug_dir}" \
  --df "${df}" \
  --up "${up}"

# Without Mahalabobis
python '../../augment.py' \
  --gold_file "${data_dir}/train.jsonl" \
  --prob_file "${out_dir}/train.prob.npy" \
  --aug_dir "${aug_dir}" \
  --up "${up}"
