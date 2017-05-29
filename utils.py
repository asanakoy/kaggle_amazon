import numpy as np
from sklearn.metrics import fbeta_score

def get_label_maps(df_train):
    flatten = lambda l: [item for sublist in l for item in sublist]
    unique_labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for lbl, idx in label_map.items()}
    return label_map, inv_label_map


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
