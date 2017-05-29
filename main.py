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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import keras
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from kerasext import create_network
from kerasext import get_preprocess_input_fn
from utils import get_label_maps
from utils import f2_score_at_02
from utils import f2_score_at_03
from utils import f2_score_at_05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suf', help='Checkpoints dir suffix.', default='')
    parser.add_argument('--model', default=None, help='Network model type.',
                        choices=['vgg16', 'vgg19', 'inception_v3', 'resnet_v2_50',
                                 'resnet_v2_101', 'resnet_v2_152', 'inception_resnet_v2'])
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate.')

    parser.add_argument('--val_part', type=float, default=0.2,
                        help='part of the validation split')

    parser.add_argument('--tile_size', type=int, default=224,
                        help='part of the validation split')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    # parser.add_argument('--num_layers_to_fix', type=int, default=0,
    #                     help='number of layers to fix')

    parser.add_argument('--zoom_range', type=float, default=0.15,
                        help='zoom range')

    parser.add_argument('--dbg', action='store_true',
                        help='Should debug (load only few images)?')
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    args = parse_args()
    print 'Args:', args
    model_dir = join('/export/home/asanakoy/workspace/kaggle/amazon/checkpoints', args.model + args.suf)
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
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=args.val_part,
                                                          random_state=42)
    gc.collect()

    initial_epoch = 0
    # Load model from model_dir directory if exists.
    # Otherwise create new model and path to the model.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        model = create_network(args.model, tile_size=args.tile_size, lr=args.lr)
    else:
        if len(os.listdir(model_dir)) == 0:
            model = create_network(args.model, tile_size=args.tile_size, lr=args.lr)
        else:
            # find the latest model
            list_of_models = glob.glob1(model_dir, '*.hdf5')
            ckpt_epochs = [int(x.split('-')[-2]) for x in list_of_models]
            print ckpt_epochs
            latest_model_name = list_of_models[np.argsort(ckpt_epochs)[-1]]
            initial_epoch = int(latest_model_name.split('-')[-2])
            print 'restoring from last snapshot:', latest_model_name
            print 'initial_epoch=', initial_epoch
            model = load_model(os.path.join(model_dir, latest_model_name))

    model_out_path = join(model_dir, 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(model_out_path, monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='auto', period=1)
    checkpointer.epochs_since_last_save = initial_epoch
    initial_epoch = int(initial_epoch)
    print("Epochs since last save: %d." % checkpointer.epochs_since_last_save)

    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=None,
        shear_range=0.0,
        zoom_range=args.zoom_range,
        horizontal_flip=True,
        fill_mode='reflect')

    train_generator = datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    # val_generator = datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=True)

    model.fit_generator(train_generator,
                        steps_per_epoch=len(x_train) // args.batch_size // 2,
                        epochs=10000,
                        initial_epoch=initial_epoch,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        max_q_size=10,
                        workers=4,
                        callbacks=[TensorBoard(log_dir=model_dir), checkpointer],
                        )

    p_valid = model.predict(x_valid, batch_size=args.batch_size)
    print(y_valid)
    print(p_valid)
    print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

