from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import tensorflow as tf

import sys; sys.path.append('../cleverhans')
from cleverhans.utils_tf import train, model_eval, model_loss, tf_model_load
from cleverhans.utils import set_log_level
from cleverhans.loss import LossCrossEntropy
import scipy as sc

from models import make_simple_cnn, make_resnet, make_vgg16

set_log_level(logging.DEBUG)
tf.set_random_seed(1822)

def load_cifar10(augmented=False):
    import keras
    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
                        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if augmented:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)
        return datagen, (x_train, y_train), (x_test, y_test)
    else:
        return (x_train, y_train), (x_test, y_test)

def train_cifar10_classifier(model_name, nb_epochs):
    rng = np.random.RandomState([2018, 8, 7])

    (x_train, y_train), (x_test, y_test) = load_cifar10(augmented=False)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    keep_prob = tf.placeholder(tf.float32, ())
    
    if model_name == 'simple':
        model = make_simple_cnn(keep_prob=keep_prob)
        train_params = {
            'nb_epochs': nb_epochs,
            'batch_size': 128,
            'learning_rate': 1e-3}
        eval_params = {'batch_size': 128}
    elif model_name == 'resnet':
        model = make_resnet(depth=32)
        train_params = {
            'nb_epochs': nb_epochs,
            'batch_size': 32,
            'learning_rate': 1e-3}
        eval_params = {'batch_size': 32}
    assert len(model.get_params()) == len(tf.trainable_variables())

    preds = model.get_probs(x)
    loss = LossCrossEntropy(model, 0)

    def evaluate():
        acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params, feed={keep_prob: 1.0})
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    sess = tf.Session()
    train(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
          args=train_params, feed={keep_prob: 0.5}, rng=rng,
          var_list=model.get_params())

    savedir = '../tfmodels'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    model_savename = 'cifar10_%s_model_epoch%d' % (model_name, nb_epochs)
    saver.save(sess, os.path.join(savedir, model_savename))

def validate_model(sess, x, y, model):
    '''
    Make sure the model load properly by running it against the test set
    '''
    (x_train, y_train), (x_test, y_test) = load_cifar10(augmented=False)
    predictions = model.get_probs(x)
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, predictions, X_test=x_test, Y_test=y_test, args=eval_params)
    print('Base accuracy of the target model on legitimate images: ' + str(accuracy))

def attack_classifier(sess, x, y, model, x_test, y_test, attack_method="fgsm", target=None, batch_size=128):

if __name__ == '__main__':
    #train_cifar10_classifier('simple', 50)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model = make_simple_cnn()
    sess = tf.Session()

    tf_model_load(sess, '../tfmodels/cifar10_simple_model_epoch50')

    validate_model(sess, x, y, model)