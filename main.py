# -*- coding: utf-8 -*-
import argparse
import cv2
import sys
import os
from os.path import join
import gc
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import keras
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from kerasext import create_network
from kerasext import get_preprocess_input_fn
from kerasext import MyLearningRateScheduler
from kerasext import MyTensorBoard
from utils import get_label_maps
from utils import f2_score_at_02
from utils import f2_score_at_03
from utils import f2_score_at_05
from utils import get_last_checkpoint

from custom_metrics import f2score_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suf', help='Checkpoints dir suffix.', default='')
    parser.add_argument('--model', default=None, help='Network model type.',
                        choices=['vgg16', 'vgg19', 'inception_v3', 'resnet50', 'resnet_v2_50',
                                 'resnet_v2_101', 'resnet_v2_152', 'inception_resnet_v2'])
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
    parser.add_argument('--end_lr', default=0.0001, type=float, help='learning rate.')

    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number [0-4]')

    parser.add_argument('--tile_size', type=int, default=224,
                        help='part of the validation split')

    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='batch_size')

    # parser.add_argument('--num_layers_to_fix', type=int, default=0,
    #                     help='number of layers to fix')

    parser.add_argument('--zoom_range', type=float, default=0.2,
                        help='zoom range')

    parser.add_argument('--dbg', action='store_true',
                        help='Should debug (load only few images)?')
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    args = parse_args()
    print 'Args:', args
    model_dir = join('/export/home/asanakoy/workspace/kaggle/amazon/checkpoints', args.model + args.suf + '_fold{}'.format(args.fold))
    print('model =', args.model)
    print('model_dir =', model_dir)
    print('batch_size =', args.batch_size)

    x_train = []
    x_test = []
    y_train = []

    df_train = pd.read_csv('../input/train_v2.csv', index_col='image_name')
    label_map, inv_label_map = get_label_maps(df_train)

    if args.dbg:
        df_train = df_train[:256]

    for row in tqdm(df_train.itertuples(), total=len(df_train)):
        img = cv2.imread('../input/train-jpg/{}.jpg'.format(row.Index))
        assert img is not None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        targets = np.zeros(17, dtype=np.uint8)
        for t in row.tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, (args.tile_size, args.tile_size)))
        y_train.append(targets)

    y_train = np.asarray(y_train, np.uint8)
    x_train = np.asarray(x_train, np.float32)
    x_train = get_preprocess_input_fn(args.model)(x_train)

    print(x_train.shape)
    print(y_train.shape)

    n_folds = 5
    kfold = KFold(n_splits=5, random_state=2002, shuffle=True)
    folds = list(kfold.split(x_train))

    assert 0 <= args.fold < n_folds
    folds = folds[args.fold:args.fold + 1]
    for train_index, valid_index in folds:

        x_train_fold, x_valid_fold = x_train[train_index], x_train[valid_index]
        y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]

        gc.collect()
        initial_epoch = 0
        # Load model from model_dir directory if exists.
        # Otherwise create new model and path to the model.
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if len(glob.glob1(model_dir, '*.hdf5')) == 0:
            model = create_network(args.model, tile_size=args.tile_size,
                                   lr=args.lr)
        else:
            checkpoint_path, initial_epoch = get_last_checkpoint(model_dir)
            try:
                model = load_model(checkpoint_path, custom_objects={'f2score_samples': f2score_samples})
            except:
                print 'Loading weights'
                model = create_network(args.model, tile_size=args.tile_size,
                                       lr=args.lr)
                model.load_weights(checkpoint_path)

        model_out_path = join(model_dir, 'checkpoint-{epoch:02d}-{val_loss:.3f}.hdf5')
        checkpointer = ModelCheckpoint(model_out_path, monitor='val_loss', verbose=1,
                                       save_best_only=True, mode='auto', period=1)
        checkpointer.epochs_since_last_save = initial_epoch
        initial_epoch = int(initial_epoch)
        print("Epochs since last save: %d." % checkpointer.epochs_since_last_save)

        datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=None,
            shear_range=0.0,
            zoom_range=args.zoom_range,
            horizontal_flip=True,
            fill_mode='reflect')

        lr_scheduler = MyLearningRateScheduler(epoch_unfreeze=80, start_lr=args.lr, end_lr=args.end_lr)

        train_generator = datagen.flow(x_train_fold, y_train_fold, batch_size=args.batch_size, shuffle=True)
        # val_generator = datagen.flow(x_train_fold, y_train_fold, batch_size=args.batch_size, shuffle=True)

        model.fit_generator(train_generator,
                            steps_per_epoch=len(x_train_fold) // args.batch_size // 2,
                            epochs=200,
                            initial_epoch=initial_epoch,
                            verbose=1,
                            validation_data=(x_valid_fold, y_valid_fold),
                            max_q_size=10,
                            workers=4,
                            callbacks=[lr_scheduler, MyTensorBoard(log_dir=model_dir), checkpointer],
                            )

        p_valid_fold = model.predict(x_valid_fold, batch_size=args.batch_size)
        print(y_valid_fold)
        print(p_valid_fold)
        print(fbeta_score(y_valid_fold, np.array(p_valid_fold) > 0.2, beta=2, average='samples'))
