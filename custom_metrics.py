from keras import backend as K
import numpy as np

from sklearn.metrics import fbeta_score

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def f2score_samples(y_true, y_pred, thresh=0.2):
    y_pred = K.round(K.clip(y_pred + thresh, 0, 1))
    return fscore(y_true, y_pred, average='samples', beta=2)


def fscore(y_true, y_pred, average='samples', beta=2):
    sum_axis = 1 if average == 'samples' else 0

    # calculate weighted counts
    true_and_pred = K.round(K.clip(y_true * y_pred, 0, 1))
    tp_sum = K.sum(true_and_pred, axis=sum_axis)
    pred_sum = K.sum(y_pred, axis=sum_axis)
    true_sum = K.sum(y_true, axis=sum_axis)

    beta2 = beta ** 2

    precision = tp_sum / (pred_sum + K.epsilon())
    recall = tp_sum / (true_sum + K.epsilon())

    f_score = ((1 + beta2) * precision * recall /
               (beta2 * precision + recall + K.epsilon()))
    # f_score[tp_sum == 0] = 0.0
    # f_score = K.switch(K.equal(f_score, 0.0), 0.0, f_score)
    return K.mean(f_score)
