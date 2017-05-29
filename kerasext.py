from os.path import join
import numpy as np
import cv2
from tqdm import tqdm
import keras
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
import keras.applications
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics


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


def get_preprocess_input_fn(model_name):
    print 'get_preprocess_input_fn({})'.format(model_name)
    return getattr(keras.applications, model_name).preprocess_input


def get_network_fn(model_name):
    model_fns = {
        'vgg19': create_vgg19,
        'inception_v3': create_inception_v3,
    }
    return model_fns[model_name]


def create_network(model_name, tile_size=224, lr=1e-6):
    input_img = Input(shape=(tile_size, tile_size, 3))
    net = get_network_fn(model_name)(input_img)
    net = Dropout(0.5)(net)

    predicts = Dense(17, activation='sigmoid')(net)
    model = Model(input_img, [predicts])

    adam = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

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
