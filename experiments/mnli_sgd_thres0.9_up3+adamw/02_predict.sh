#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --out=predict.%A.log
#SBATCH --time=00:10:00
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

# Remove all the suffixes until the "_" character
orig_data_name="${data_name%%_*}"
data_dir="../data/${orig_data_name}"

pretrained='bert-base-uncased'
max_len=128
model_dir="${pretrained}-${max_len}-mod"
out_dir="${pretrained}-${max_len}-out"

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${latest}" ]] && latest="${file}"
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

mkdir -p "${out_dir}"

for split in 'dev' 'test'; do
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python '../../predict.py' \
    --checkpoint_file "${latest}" \
    --in_file "${data_dir}/${split}.jsonl" \
    --out_file "${out_dir}/${split}.prob.npy" \
    --batch_size 128 \
    --gpus 1

  python ../../evaluate_classes.py \
    --gold_file "${data_dir}/${split}.jsonl" \
    --prob_file "${out_dir}/${split}.prob.npy" \
    --out_file "${out_dir}/eval.${split}.txt"

  python ../../evaluate_groups.py \
    --gold_file "${data_dir}/${split}.jsonl" \
    --prob_file "${out_dir}/${split}.prob.npy" \
    --out_file "${out_dir}/eval.groups.${split}.txt"
done
