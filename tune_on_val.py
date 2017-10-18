# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import sys
import os
from os.path import join
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from keras.models import load_model
from pprint import pprint
from scipy.misc import imread

from utils import get_label_maps
from utils import tags_to_one_hot
import kerasext
from kerasext import predict_tta
from args_utils import parse_args_predict
from args_utils import get_pathes_predict
import predict as predict_script
from custom_metrics import f2score_samples


if __name__ == '__main__':
    args = parse_args_predict()
    split_for_threshold_tuning = 'all_train' # 'val
    print 'Args:', args

    images_dir, model_dir, pred_out_dir, probs_path, checkpoint_path = \
        get_pathes_predict(args, split=split_for_threshold_tuning)
    print 'probs_path', probs_path

    df_train_orig = pd.read_csv('../input/train_v2.csv', index_col='image_name')
    label_map, inv_label_map = get_label_maps(df_train_orig)

    if split_for_threshold_tuning == 'val':
        n_folds = 5
        kfold = KFold(n_splits=5, random_state=2002, shuffle=True)
        folds = list(kfold.split(df_train_orig.values))

        assert 0 <= args.fold < n_folds
        train_index, valid_index = folds[args.fold]
    else:
        assert split_for_threshold_tuning == 'all_train'
        valid_index = np.arange(len(df_train_orig))

    df_val = df_train_orig.iloc[valid_index]

    y_val = tags_to_one_hot(df_val, label_map)
    num_train = len(df_val)

    if False and os.path.exists(probs_path):
        predicts = np.load(probs_path)
        model = None
    else:
        model = load_model(checkpoint_path, custom_objects={'f2score_samples': f2score_samples})
        predicts = list()
        num_blocks = 40
        block_size = int(np.ceil(len(df_val) / float(num_blocks)))
        for block_start in xrange(0, len(df_val), block_size):
            print 'Block', block_start / block_size
            block_df = df_val.iloc[block_start:block_start + block_size]
            block_predicts = predict_tta(args.model, model,
                                         images_dir, block_df.index,
                                         batch_size=args.batch_size,
                                         crop_size=args.tile_size,
                                         n_augs=args.n_tta)
            predicts.append(block_predicts)

        predicts = np.vstack(predicts)
        np.save(probs_path, predicts)
    print 'Predictions shape:', predicts.shape
    print predicts[:5]

    # predicts is np.array of probabilities Nx17
    results = defaultdict(dict)
    best_thresh = dict()
    best_score = {x: 0 for x in xrange(17)}
    thresholds = np.array([0.2] * 17)
    for class_id in xrange(17):
        for thresh in tqdm(np.linspace(0, 1.0, num=101, dtype=float)):
            cur_thresholds = thresholds.copy()
            cur_thresholds[class_id] = thresh
            bin_preds = predicts > cur_thresholds
            score = fbeta_score(y_val, bin_preds, beta=2, average='samples')
            results[class_id][thresh] = score
            if score > best_score[class_id]:
                best_score[class_id] = score
                best_thresh[class_id] = thresh
        thresholds[class_id] = best_thresh[class_id]

    pprint(results)
    print '===================='
    pprint(best_thresh)
    pprint(best_score)

    assert np.all(thresholds == np.array([best_thresh[class_id] for class_id in xrange(17)]))

    bin_preds = np.zeros_like(predicts, dtype=np.uint8)
    for i, probs in tqdm(enumerate(predicts), desc='imputing labels'):
        bin_preds[i, ...] = probs > thresholds
        # if not len(indices):
        #     indices = [np.argmax(probs)]
    for thresh in np.linspace(0, 1.0, num=11, dtype=float):
        print 'val score with thresh={}: {}'.format(thresh, fbeta_score(y_val, predicts > thresh, beta=2, average='samples'))
    score = fbeta_score(y_val, bin_preds, beta=2, average='samples')
    print 'Final val score (thresh={}): {}'.format(thresholds, score)
    np.save(join(pred_out_dir, '{}_cv_thresholds_{}_tta{}{}.npy'.format(
                                                                    split_for_threshold_tuning,
                                                                    os.path.splitext(args.checkpoint)[0],
                                                                   args.n_tta, args.pred_suf)), thresholds)
    print 'Thresholds saved'

    predict_script.main(args, model=model, thresholds=thresholds, split_for_threshold_tuning=split_for_threshold_tuning)
