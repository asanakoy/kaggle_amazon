#!/usr/bin/env bash

 for fold_id in 0 1 2 3 4
 do
     echo "FOLD ${fold_id}"
     python tune_on_val.py --model=resnet50 --batch_size=128 --suf="_lr0.0001_fold${fold_id}" --n_tta=12 --pred_suf=_threshsteps101
 done