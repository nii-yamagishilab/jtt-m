#!/bin/bash

set -ex

# Download datasets and uncompress
url='https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc_baselines'
for name in 'fever' 'mnli'; do
  if [[ ! -f "${name}.zip" ]]; then
    wget "${url}/${name}.zip"
  fi

  if [[ ! -d "${name}" ]]; then
    unzip "${name}.zip"
  fi

  wc -l "${name}"/{train,dev,test}.jsonl
  python '../../group_stats.py' "${name}"/train.jsonl
done
