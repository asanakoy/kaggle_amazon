#!/usr/bin/env bash

#python tune_on_val.py --model=resnet50 --batch_size=128 --suf="_lr0.0001_fold${1}" --n_tta=12

for fold_id in 0 1 # 3
do
    gpu=$((fold_id + 4))
    echo "fold ${fold_id}; GPU ${gpu}"
    tmux \
        new-window "CUDA_VISIBLE_DEVICES=${gpu} python tune_on_val.py --model=resnet50 --batch_size=128 --suf="_lr0.0001_fold${fold_id}" --n_tta=18 --fold=${fold_id}" \;
done
