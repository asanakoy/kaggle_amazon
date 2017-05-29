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

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from keras.models import load_model
from pprint import pprint
from scipy.misc import imread

from utils import get_label_maps
from utils import tags_to_one_hot
import kerasext
from data_utils import parse_args_predict
from data_utils import get_pathes_predict
import predict as predict_script

if __name__ == '__main__':
    args = parse_args_predict()
    print 'Args:', args

    images_dir, model_dir, pred_out_dir, probs_path, checkpoint_path = \
        get_pathes_predict(args, split='val')
    print 'probs_path', probs_path

    df_train = pd.read_csv('../input/train_v2.csv', index_col='image_name')
    label_map, inv_label_map = get_label_maps(df_train)
    df_train, df_val = train_test_split(df_train,
                                        test_size=args.val_part,
                                        random_state=42)

    y_val = tags_to_one_hot(df_val, label_map)
    num_train = len(df_val)

    if os.path.exists(probs_path):
        predicts = np.load(probs_path)
        model = None
    else:
        model = load_model(checkpoint_path)
        predicts = list()
        num_blocks = 2
        block_size = int(np.ceil(len(df_val) / float(num_blocks)))
        for block_start in xrange(0, len(df_val), block_size):
            print 'Block', block_start / block_size
            block_df = df_val.iloc[block_start:block_start + block_size]
            block_predicts = kerasext.predict(args.model, model, images_dir, block_df.index, args.batch_size)
            predicts.append(block_predicts)

        predicts = np.vstack(predicts)
        np.save(probs_path, predicts)
    print 'Predictions shape:', predicts.shape
    print predicts[:5]

    results = defaultdict(dict)
    best_thresh = dict()
    best_score = {x: 0 for x in xrange(17)}
    thresholds = np.array([0.2] * 17)
    for class_id in xrange(17):
        for thresh in tqdm(np.linspace(0, 1.0, num=21, dtype=float)):
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
    np.save(join(pred_out_dir, 'cv_thresholds.npy'), thresholds)
    print 'Thresholds saved'

    predict_script.main(args, model=model, thresholds=thresholds)
