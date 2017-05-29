# -*- coding: utf-8 -*-
import argparse
import cv2
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
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from data_utils import get_pathes_predict
from data_utils import parse_args_predict
from data_utils import ROOT_DIR
from utils import get_label_maps
from kerasext import predict


def main(args, model=None, thresholds=None):
    print '=== Predict on test ==='
    images_dir, model_dir, pred_out_dir, probs_path, checkpoint_path = \
        get_pathes_predict(args, split='test')

    df_train = pd.read_csv('../input/train_v2.csv', index_col='image_name')
    label_map, inv_label_map = get_label_maps(df_train)
    df_sample_test = pd.read_csv('../input/sample_submission_v2.csv', index_col='image_name')

    if os.path.exists(probs_path):
        predicts = np.load(probs_path)
    else:
        if not model:
            model = load_model(checkpoint_path)
        predicts = list()
        num_blocks = 10
        block_size = int(np.ceil(len(df_sample_test) / float(num_blocks)))
        for block_start in xrange(0, len(df_sample_test), block_size):
            print 'Block', block_start / block_size
            block_df = df_sample_test.iloc[block_start:block_start + block_size]
            block_predicts = predict(args.model, model,
                                     images_dir, block_df.index,
                                     batch_size=args.batch_size,
                                     tile_size=args.tile_size)
            predicts.append(block_predicts)

        predicts = np.vstack(predicts)
        np.save(probs_path, predicts)
    print 'Predictions shape:', predicts.shape
    print predicts[:5]

    cnt = 0
    if thresholds is None:
        thresholds = np.load(join(pred_out_dir, 'cv_thresholds.npy'))
    assert len(thresholds) == 1 or len(thresholds) == predicts.shape[1], \
        'wrong thresholds shape:{}'.format(thresholds.shape)
    print 'Cur thresholds:', thresholds
    # thresholds = 0.1
    predicted_labels = list()
    # TODO: make weather label self excluding
    for probs in tqdm(predicts, desc='imputing labels'):
        indices = np.nonzero(probs > thresholds)[0]
        if not len(indices):
            cnt += 1
            indices = [np.argmax(probs)]
        cur_labels = ' '.join([inv_label_map[i] for i in indices])
        predicted_labels.append(cur_labels)

    print 'Were Without pred:', cnt
    final_df = pd.DataFrame(index=df_sample_test.index, data={'tags': predicted_labels})
    final_df.index.name = 'image_name'
    final_df.head()
    final_df.to_csv(join(ROOT_DIR, 'predictions/submission_df.csv'))


if __name__ == '__main__':
    args = parse_args_predict()
    print 'Args:', args
    main(args)
