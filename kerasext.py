from os.path import join
import numpy as np
import cv2
from tqdm import tqdm
import keras
from keras import backend as K
from keras.layers.core import Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import keras.applications
from keras.layers.core import Lambda
import tensorflow as tf

import custom_metrics


def create_vgg19(input_img):
    net_template = VGG19(weights='imagenet', include_top=True)
    x = input_img
    for layer in net_template.layers[1:5]:
        layer.trainable = False
        x = layer(x)
    for layer in net_template.layers[5:-1]:
        layer.trainable = True
        x = layer(x)
    return x


def create_inception_v3(input_img):
    net = InceptionV3(weights='imagenet', include_top=False,
                      input_tensor=input_img, pooling='avg')
    # x = input_img
    for layer in net.layers[1:11]:
        layer.trainable = False
    return net.outputs[0]


def create_resnet50(input_img):
    net = ResNet50(weights='imagenet', include_top=False,
                      input_tensor=input_img)

    for layer in net.layers[1:]:
        layer.trainable = False
    net = Reshape((-1,))(net.outputs[0])
    return net


def get_preprocess_input_fn(model_name):
    print 'get_preprocess_input_fn({})'.format(model_name)
    return getattr(keras.applications, model_name).preprocess_input


def get_network_fn(model_name):
    model_fns = {
        'vgg19': create_vgg19,
        'inception_v3': create_inception_v3,
        'resnet50': create_resnet50
    }
    return model_fns[model_name]


def create_network(model_name, tile_size=224, lr=1e-6):
    input_img = Input(shape=(tile_size, tile_size, 3))
    net = get_network_fn(model_name)(input_img)
    net = Dropout(0.5)(net)

    predicts = Dense(17, activation='sigmoid')(net)
    model = Model([input_img], [predicts])

    adam = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy',
                                                                       custom_metrics.f2score_samples])

    tf.summary.scalar('lr', model.optimizer.lr)

    print(model.summary())
    return model


def predict(model_name, model, images_dir, image_ids, batch_size=64, tile_size=224):
    x_test = np.zeros((len(image_ids), tile_size, tile_size, 3), dtype=np.float32)

    for idx, image_name in tqdm(enumerate(image_ids), total=len(image_ids)):
        # img = imread(join(images_dir, '{}.jpg'.format(image_name)))
        image_path = join(images_dir, '{}.jpg'.format(image_name))
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(cv2.resize(img, (tile_size, tile_size)), dtype=np.float32)
            x_test[idx, ...] = img
        except Exception as e:
            print e.message
            print 'image:', image_path
    x_test = get_preprocess_input_fn(model_name)(x_test)
    print(x_test.shape)
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    return predictions


def predict_tta(model_name, model, images_dir, image_ids, batch_size=64, crop_size=224, n_augs=2):
    print 'Using {} Test Time Augs'.format(n_augs)
    add_full_image = True
    if n_augs == 1:
        n_crops = 0
        fliplr = False
        flipud = False
    elif n_augs == 2:
        n_crops = 1
        fliplr = False
        flipud = False
    elif n_augs == 12:
        n_crops = 5
        fliplr = True
        flipud = False
    elif n_augs == 18:
        n_crops = 5
        fliplr = True
        flipud = True
    else:
        raise ValueError('Wrong n_augs: {}'.format(n_augs))

    img_size = int(crop_size * 1.143)
    x_test = np.zeros((len(image_ids) * n_augs, crop_size, crop_size, 3), dtype=np.float32)

    for idx, image_name in tqdm(enumerate(image_ids), total=len(image_ids)):
        # img = imread(join(images_dir, '{}.jpg'.format(image_name)))
        image_path = join(images_dir, '{}.jpg'.format(image_name))
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(cv2.resize(img, (img_size, img_size)))
            x_test[idx * n_augs: (idx + 1) * n_augs, ...] = tta(img,
                                                                add_full_image=add_full_image,
                                                                n_crops=n_crops,
                                                                crop_size=crop_size,
                                                                fliplr=fliplr,
                                                                flipud=flipud)
        except Exception as e:
            print e.message
            print 'image:', image_path
    x_test = get_preprocess_input_fn(model_name)(x_test)
    print(x_test.shape)
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    assert len(predictions) == len(image_ids) * n_augs

    avg_predictionts = list()
    for idx in xrange(0, len(predictions), n_augs):
        avg_predictionts.append(np.average(predictions[idx:idx + n_augs, :], axis=0))
    predictions = np.vstack(avg_predictionts)
    assert len(predictions) == len(image_ids)
    return predictions


def tta(image, n_crops=5, crop_size=224, add_full_image=True, fliplr=True, flipud=False):
    """
    Generate deterministic augmentation for an image (test time augmentations)
    Args:
        image:
        n_crops:
        crop_size:
        add_full_image: add full image and resized to crop_size?
        fliplr:
        flipud:

    Returns:

    """
    n_augs = (n_crops + add_full_image) * (1 + fliplr + flipud)
    assert image.shape[-1] == 3, image.shape
    augmented_images = np.zeros((n_augs, crop_size, crop_size, image.shape[-1]), image.dtype)

    idx = 0

    if add_full_image:
        augmented_images[idx, ...] = cv2.resize(image, (crop_size, crop_size))
        idx += 1

    assert (image.shape[0] - crop_size) % 2 == 0, (image.shape[0] - crop_size)
    offset = (image.shape[0] - crop_size) // 2
    offsets = [(offset, offset),  # central crop
               (0, 0), (0, 2 * offset),
               (2 * offset, 2 * offset), (2 * offset, 0)][:n_crops]

    for offset_row, offset_col in offsets:
        augmented_images[idx, ...] = image[offset_row:offset_row + crop_size,
                                     offset_col:offset_col + crop_size, ...]
        idx += 1

    cur_num_augs = idx
    for i in xrange(cur_num_augs):
        if fliplr:
            augmented_images[idx, ...] = np.fliplr(augmented_images[i, ...])
            idx += 1
        if flipud:
            augmented_images[idx, ...] = np.flipud(augmented_images[i, ...])
            idx += 1
    assert idx == len(augmented_images), '{} != {}'.format(idx, len(augmented_images))
    return augmented_images


class MyLearningRateScheduler(LearningRateScheduler):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, epoch_unfreeze, start_lr, end_lr=1e-4, num_layers_to_freeze=10):
        self.epoch_unfreeze = epoch_unfreeze
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_layers_to_freeze = num_layers_to_freeze
        self.recompiled_first = False
        self.recompiled = False
        super(MyLearningRateScheduler, self).__init__(self.step_decay)

    def step_decay(self, epoch):
        if epoch > self.epoch_unfreeze:
            lr = self.end_lr
        else:
            lr = self.start_lr
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        super(MyLearningRateScheduler, self).on_epoch_begin(epoch, logs=logs)

        if epoch > self.epoch_unfreeze:
            for i, layer in enumerate(self.model.layers[1:]):
                layer.trainable = i >= self.num_layers_to_freeze
        else:
            for layer in self.model.layers[1:-1]:
                layer.trainable = False
            self.model.layers[-1].trainable = True

        if not self.recompiled_first or (not self.recompiled and epoch > self.epoch_unfreeze):
            adam = keras.optimizers.Adam(lr=self.step_decay(epoch))
            self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy',
                                                                                    custom_metrics.f2score_samples])
            print self.model.summary()
            if not self.recompiled_first:
                self.recompiled_first = True
            else:
                self.recompiled = True


class MyTensorBoard(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)
        else:
            summary_str = self.sess.run(self.merged)
            self.writer.add_summary(summary_str, epoch)

        if self.embeddings_freq and self.embeddings_logs:
            if epoch % self.embeddings_freq == 0:
                for log in self.embeddings_logs:
                    self.saver.save(self.sess, log, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()


# def make_parallel(model, gpu_count):
#     def get_slice(data, idx, parts):
#         shape = tf.shape(data)
#         stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
#         start = stride * idx
#         size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
#         size = tf.minimum(shape - start, size)
#
#         # size = [1, 224, 224, 3]
#         slice = tf.slice(data, start, size)
#         slice = tf.Print(slice, [tf.shape(slice)], 'batch Size:', summarize=5)
#         return slice
#
#     outputs_all = []
#     for i in range(len(model.outputs)):
#         outputs_all.append([])
#
#     # Place a copy of the model on each GPU, each getting a slice of the batch
#     for i in range(gpu_count):
#         with tf.device('/gpu:%d' % i):
#             with tf.name_scope('tower_%d' % i) as scope:
#
#                 inputs = []
#                 # Slice each input into a piece for processing on this GPU
#                 for x in model.inputs:
#                     input_shape = tuple(x.get_shape().as_list())[1:]
#                     slice_n = Lambda(get_slice, output_shape=input_shape,
#                                      arguments={'idx': i, 'parts': gpu_count})(x)
#                     inputs.append(slice_n)
#
#                 outputs = model(inputs)
#
#                 if not isinstance(outputs, list):
#                     outputs = [outputs]
#
#                 # Save all the outputs for merging back together later
#                 for l in range(len(outputs)):
#                     outputs_all[l].append(outputs[l])
#
#     # merge outputs on CPU
#     with tf.device('/cpu:0'):
#         merged = []
#         for outputs in outputs_all:
#             merged.append(keras.layers.Concatenate(axis=0)(outputs))
#
#         return Model(inputs=model.inputs, outputs=merged)
