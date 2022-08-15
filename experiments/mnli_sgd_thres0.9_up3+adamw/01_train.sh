#!/bin/bash
#SBATCH --job-name=train
#SBATCH --out=train.%A.log
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate jtt-m
fi

set -ex

echo "${HOSTNAME}"
base_dir="$(basename "$PWD")"

# Remove suffix until the "+" character
data_name="${base_dir%+*}"
data_dir="../data/${data_name}"

# Remove prefix until the "+" character
loss_fn="${base_dir#*+}"

pretrained='bert-base-uncased'
max_len=128
model_dir="${pretrained}-${max_len}-mod"

if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python '../../train.py' \
  --data_dir "${data_dir}" \
  --default_root_dir "${model_dir}" \
  --pretrained_model_name "${pretrained}" \
  --loss_fn "${loss_fn}" \
  --max_seq_length "${max_len}" \
  --max_epochs 2 \
  --skip_validation \
  --cache_dir "/local/$(whoami)" \
  --overwrite_cache \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --accumulate_grad_batches 1 \
  --learning_rate 2e-5 \
  --gradient_clip_val 1.0 \
  --precision 16 \
  --deterministic true \
  --gpus 1
