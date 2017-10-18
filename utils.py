import glob
import numpy as np
import pandas as pd
import os
from os.path import join
from sklearn.metrics import fbeta_score
from tqdm import tqdm

from data_utils import ROOT_DIR


def get_label_maps(df_train):
    flatten = lambda l: [item for sublist in l for item in sublist]
    unique_labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for lbl, idx in label_map.items()}
    return label_map, inv_label_map


def labels_df_2_binary_df(df):
    df_train = pd.read_csv(join(ROOT_DIR, 'input/train_v2.csv'), index_col='image_name')
    label_map, inv_label_map = get_label_maps(df_train)

    target_matrix = np.zeros((len(df), 17), dtype=np.uint8)
    for i, row in tqdm(enumerate(df.itertuples()), total=len(df_train)):
        targets = np.zeros(17, dtype=np.uint8)
        for t in row.tags.split(' '):
            targets[label_map[t]] = 1
        target_matrix[i, :] = targets

    binary_df = pd.DataFrame(index=df.index,
                             data=target_matrix,
                             columns=[inv_label_map[class_id] for class_id in xrange(17)])
    return binary_df


def binary_df_2_labels_df(df):
    df_train = pd.read_csv(join(ROOT_DIR, 'input/train_v2.csv'), index_col='image_name')
    label_map, inv_label_map = get_label_maps(df_train)

    predicted_labels = list()
    # TODO: make weather label self excluding
    for binary_vector in tqdm(df.values):
        indices = np.nonzero(binary_vector)[0]
        cur_labels = ' '.join([inv_label_map[i] for i in indices])
        predicted_labels.append(cur_labels)
    label_df = pd.DataFrame(index=df.index, data={'tags': predicted_labels})
    return label_df


def tags_to_one_hot(df, label_map):
    targets = np.zeros((len(df), 17), dtype=np.uint8)
    for i, tags in enumerate(df['tags']):
        for t in tags.split(' '):
            targets[i, label_map[t]] = 1
    return targets


def f2_score(y_true, y_preds):
    return fbeta_score(y_true, y_preds, beta=2, average='samples')


def f2_score_at(y_true, y_probs, thresh=0.5):
    y_preds = y_probs >= thresh
    return f2_score(y_true, y_preds)


def f2_score_at_05(y_true, y_probs):
    return f2_score_at(y_true, y_probs, thresh=0.5)


def f2_score_at_03(y_true, y_probs):
    return f2_score_at(y_true, y_probs, thresh=0.3)


def f2_score_at_02(y_true, y_probs):
    return f2_score_at(y_true, y_probs, thresh=0.2)


def get_last_checkpoint(model_dir):
    list_of_models = glob.glob1(model_dir, '*.hdf5')
    ckpt_epochs = [int(x.split('-')[-2]) for x in list_of_models]
    print ckpt_epochs
    latest_model_name = list_of_models[np.argsort(ckpt_epochs)[-1]]
    epoch_num = int(latest_model_name.split('-')[-2])
    print 'last snapshot:', latest_model_name
    print 'last epoch=', epoch_num
    return os.path.join(model_dir, latest_model_name), epoch_num
