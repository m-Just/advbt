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

def filter_data(sess, x, y, model, x_test, y_test, target=None, eval_size=1280, opposite=False,
                labels=None):
    pred = model.get_probs(x)
    eval_single = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))

    if labels is not None:
        y_feed = labels
    else:
        y_feed = y_test

    # only take the ones that are correctly classified by the target model
    x_filtered = []
    y_filtered = []
    indices = []
    counter = 0
    for i in range(len(x_test)):
        correct = sess.run(eval_single, feed_dict={x: [x_test[i]], y: [y_feed[i]]})
        if np.argmax(y_test[i]) != target and (opposite ^ correct):
            x_filtered.append([x_test[i]])
            y_filtered.append([y_test[i]])
            indices.append(i)
            counter += 1
        if counter >= eval_size: break

    return np.concatenate(x_filtered, axis=0), np.concatenate(y_filtered, axis=0), np.array(indices)

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    train(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
          args=train_params, feed={keep_prob: 0.5}, rng=rng,
          var_list=model.get_params())

    savedir = '../tfmodels'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    model_savename = 'cifar10_%s_model_epoch%d' % (model_name, nb_epochs)
    saver.save(sess, os.path.join(savedir, model_savename))

def validate_model(sess, x, y, model, x_test, y_test):

    predictions = model.get_probs(x)
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, predictions, X_test=x_test, Y_test=y_test, args=eval_params)
    return accuracy

def attack_classifier(sess, x, model, x_test, attack_method="fgsm", target=None, batch_size=128):

    if attack_method == "fgsm":
        from cleverhans.attacks import FastGradientMethod
        params = {'eps': 8/255,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = FastGradientMethod(model, sess=sess)

    elif attack_method == "basic_iterative":
        from cleverhans.attacks import BasicIterativeMethod
        params = {'eps': 8./255,
                  'eps_iter': 1./255,
                  'nb_iter': 10,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  'ord': np.inf
                 }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = BasicIterativeMethod(model,sess = sess)

    elif attack_method == "momentum_iterative":
        from cleverhans.attacks import MomentumIterativeMethod
        params = {'eps':8/255,
                  'eps_iter':1/255,
                  'nb_iter': 10,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = MomentumIterativeMethod(model,sess = sess)

    elif attack_method == "saliency":
        from cleverhans.attacks import SaliencyMapMethod
        params = {'theta':8/255,
                  'gamma':0.1,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        assert target is None
        method = SaliencyMapMethod(model,sess = sess)

    elif attack_method == "virtual":
        from cleverhans.attacks import VirtualAdversarialMethod
        params = {'eps':8/255,
                  'num_iterations':10,
                  'xi' :1e-6,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        assert target is None
        method = VirtualAdversarialMethod(model,sess = sess)

    elif attack_method == "cw":
        from cleverhans.attacks import CarliniWagnerL2
        params = {
            "confidence":0,
            "batch_size":128,
            "learning_rate":1e-4,
            "binary_search_steps":10,
            "max_iterations":1000,
            "abort_early": True,
            "initial_const":1e-2,
            "clip_min":0,
            "clip_max":1
        }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = CarliniWagnerL2(model,sess = sess)

    elif attack_method == "elastic_net":
        from cleverhans.attacks import ElasticNetMethod
        params = {
            "fista": "FISTA",
            "beta": 0.1,
            "decision_rule":"EN",
            "confidence":0,
            "batch_size":128,
            "learning_rate":1e-4,
            "binary_search_steps":10,
            "max_iterations":1000,
            "abort_early": True,
            "initial_const":1e-2,
            "clip_min":0,
            "clip_max":1
        }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = ElasticNetMethod(model,sess = sess)

    elif attack_method == "deepfool":
        from cleverhans.attacks import DeepFool
        params = {
            "nb_candidate":10,
            "overshoot":1e-3,
            "max_iter":100,
            "nb_classes":10,
            "clip_min":0,
            "clip_max":1
        }
        assert target is None
        method = DeepFool(model,sess = sess)

    elif attack_method == "lbfgs":
        from cleverhans.attacks import LBFGS
        params = {
            'batch_size':128,
            "binary_search_steps":10,
            "max_iterations":1000,
            "initial_const":1e-2,
            'clip_min': 0.,
            'clip_max': 1.
        }
        assert target is not None
        params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = LBFGS(model,sess = sess)

    elif attack_method == "madry":
        from cleverhans.attacks import MadryEtAl
        params = {'eps':8/255,
                  'eps_iter':1/255,
                  'nb_iter':10,
                  'ord':np.inf,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = MadryEtAl(model, sess = sess)

    elif attack_method == "SPSA":
        from cleverhans.attacks import SPSA
        params = {
            'epsilon':1/255,
            'num_steps':10,
            'is_targeted':False,
            'early_stop_loss_threshold':None,
            'learning_rate':0.01,
            'delta':0.01,
            'batch_size':128,
            'spsa_iters':1,
            'is_debug':False
        }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
            params["is_targeted"] = True
        method = SPSA(model, sess = sess)

    else:
        raise ValueError("Can not recognize this attack method: %s" % attack_method)

    adv_x = method.generate(x, **params)
    num_batch = x_test.shape[0] // batch_size
    adv_imgs = []
    for i in range(num_batch):
        x_feed = x_test[i*batch_size:(i+1)*batch_size]
        #y_feed = y_test[i*batch_size:(i+1)*batch_size]

        adv_img = sess.run(adv_x, feed_dict={x: x_feed})
        adv_imgs.append(adv_img)

    adv_imgs = np.concatenate(adv_imgs, axis=0)
    return adv_imgs

def backtracking(sess, x, model, x_test, params, batch_size=128):
    from cleverhans.attacks import BasicIterativeMethod
    method = BasicIterativeMethod(model, sess=sess)

    adv_x = method.generate(x, **params)
    num_batch = x_test.shape[0] // batch_size
    adv_imgs = []
    for i in range(num_batch):
        if i + 1 == num_batch:
            x_feed = x_test[i*batch_size:]
        else:
            x_feed = x_test[i*batch_size:(i+1)*batch_size]
        adv_img = sess.run(adv_x, feed_dict={x: x_feed})
        adv_imgs.append(adv_img)

    adv_imgs = np.concatenate(adv_imgs, axis=0)
    return adv_imgs

def backtrack_v2(sess, x, model, adv_imgs, params):

    logits = model.get_logits(x)
    probs = model.get_probs(x)
    preds = tf.to_float(tf.equal(probs, tf.reduce_max(probs, axis=1, keepdims=True)))
    labels = sess.run(preds, feed_dict={x: adv_imgs})
    labels = tf.constant(labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

    grad, = tf.gradients(loss, x)
    if params['ord'] == 1:
        norm = tf.reduce_sum(tf.abs(grad), axis=(1, 2, 3)) # L1
        norm = tf.stop_gradient(norm)
        normalized_grad = tf.transpose(tf.transpose(grad, [1, 2, 3, 0]) / norm, [3, 0, 1, 2])
    elif params['ord'] == 2:
        norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=(1, 2, 3))) # L2
        norm = tf.stop_gradient(norm)
        normalized_grad = tf.transpose(tf.transpose(grad, [1, 2, 3, 0]) / norm, [3, 0, 1, 2])
    elif params['ord'] == np.inf:
        norm = tf.constant(1. / params['eps_iter'])
        grad = tf.stop_gradient(grad)
        normalized_grad = tf.sign(grad) / norm
    adv_x = x + params['step_scale'] * params['eps'] * normalized_grad

    clip_base = tf.constant(adv_imgs)
    adv_x = tf.clip_by_value(adv_x, clip_base - params['eps'], clip_base + params['eps'])
    adv_x = tf.clip_by_value(adv_x, params['clip_min'], params['clip_max'])

    for i in range(params['nb_iter']):
        adv_imgs, loss_val, grad_val, norm_val, ng_val, logits_val = sess.run(
            [adv_x, loss, grad, norm, normalized_grad, logits], feed_dict={x: adv_imgs})
        print('Recovery iteration %d: ' % (i + 1))
        print('  loss(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(loss_val), np.min(loss_val), np.max(loss_val)))
        grad_val = np.abs(grad_val)
        print('  grad(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(grad_val), np.min(grad_val), np.max(grad_val)))
        print('  norm(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(norm_val), np.min(norm_val), np.max(norm_val)))
        print('  ngra(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(ng_val), np.min(ng_val), np.max(ng_val)))
        #print(norm_val.shape)
        #n = np.argmin(norm_val)
        #for n in range(len(adv_imgs)):
        #    if norm_val[n] > 0: continue
        #    print(n)
        #    print(norm_val[n])
        #    print(logits_val[n])
        #    print(np.min(adv_imgs[n]), np.max(adv_imgs[n]))

    return adv_imgs

def backtrack_v3(sess, x, model, adv_imgs, params):

    logits = model.get_logits(x)
    probs = model.get_probs(x)
    preds = tf.to_float(tf.equal(probs, tf.reduce_max(probs, axis=1, keepdims=True)))
    labels = sess.run(preds, feed_dict={x: adv_imgs})
    labels = tf.constant(labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

    grad, = tf.gradients(loss, x)

    normalized_grad = tf.placeholder(tf.float32, [None, 32, 32, 3])
    adv_x = x + params['step_scale'] * params['eps'] * normalized_grad
    #adv_x = x + 0.01 * tf.random_normal([len(adv_imgs), 32, 32, 3], stddev=2)

    clip_base = tf.constant(adv_imgs)
    adv_x = tf.clip_by_value(adv_x, clip_base - params['eps'], clip_base + params['eps'])
    adv_x = tf.clip_by_value(adv_x, params['clip_min'], params['clip_max'])

    for i in range(params['nb_iter']):
        grad_val = sess.run(grad, feed_dict={x: adv_imgs})
        if params['ord'] == 1:
            norm_val = np.sum(np.abs(grad_val), axis=(1, 2, 3))
        elif params['ord'] == 2:
            norm_val = np.sqrt(np.sum(grad_val ** 2, axis=(1, 2, 3)))
        else:
            raise NotImplementedError()
        grad_val = np.transpose(grad_val, [1, 2, 3, 0])
        ng_val = np.transpose(grad_val / norm_val, [3, 0, 1, 2])
        adv_imgs = sess.run(adv_x, feed_dict={x: adv_imgs, normalized_grad: ng_val})

        grad_val = np.abs(grad_val)
        print('Recovery iteration %d: ' % (i + 1))
        print('  grad(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(grad_val), np.min(grad_val), np.max(grad_val)))
        print('  norm(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(norm_val), np.min(norm_val), np.max(norm_val)))
        print('  ngra(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(ng_val), np.min(ng_val), np.max(ng_val)))

    return adv_imgs

def eval_perturbation(p):
    batch_num = len(p)
    p_abs = np.abs(p)

    l_one = np.sum(p_abs) / batch_num
    l_two = np.sum(np.sqrt(np.sum(p ** 2, axis=(1, 2, 3)))) / batch_num
    l_inf = np.sum(np.max(p_abs, axis=(1, 2, 3))) / batch_num
    gavg = np.mean(p_abs)
    gmax = np.max(p_abs)
    gmin = np.min(p_abs)
    print('%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f' % (l_one, l_two, l_inf, gmax, gavg, gmin))

def eval_adv(sess, x, model, adv_imgs, y_filtered):
    #adv_imgs += (np.random.rand(*adv_imgs.shape) > 0.5) - 0.5
    adv_imgs = adv_imgs[:, :, ::-1, :]
    probs = model.get_probs(x)
    preds = tf.to_float(tf.equal(probs,
                                 tf.reduce_max(probs,
                                               axis=1, keepdims=True)))
    labels = sess.run(preds, feed_dict={x: adv_imgs})
    for i in range(10):
        cnt = np.count_nonzero(labels[:, i])
        print('Predicted as label-%d: %f' % (i, float(cnt) / len(labels)))

    logits = model.get_logits(x)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                      labels=labels)
    grad, = tf.gradients(loss, x)
    grad_sum = tf.reduce_sum(tf.abs(grad), axis=(1, 2, 3))
    loss_val, grad_val, grad_sum_val = sess.run([loss, grad, grad_sum], feed_dict={x: adv_imgs})

    img = None
    grad_sum_val = np.abs(grad_sum_val)
    print('Gradient sum (avg)=%f, (min)=%E, (max)=%f' % \
          (np.mean(grad_sum_val), np.min(grad_sum_val), np.max(grad_sum_val)))
    for i in range(len(grad_sum_val)):
        if np.count_nonzero(grad_val[i]) == 0:
        #if i in [877, 951, 1035, 1036, 1043, 1191, 1223, 1230]:
            feed_x = [adv_imgs[i]]

            print(i)
            print('Image (min)=%f, (max)=%f' % \
                  (np.min(feed_x), np.max(feed_x)))
            preds_val = sess.run(probs, feed_dict={x: feed_x})
            print(loss_val[i] == 0, loss_val[i], preds_val)
            print(grad_sum_val[i])

            if img is not None:
                print('Difference=%f' % (np.mean(np.abs(feed_x[0] - img))))
            img, = feed_x
            #print(grad_val[i])
            #print(np.argmax(preds_val))
            #print(np.argmax(labels[i]))
            #print(np.argmax(y_filtered[i]))

def attack_and_recover(sess, x, y, x_benign, y_benign, model, target, attack_method,
                       recover_method, recover_params, flip_image=False):

    adv_imgs = attack_classifier(sess, x, model, x_benign, attack_method='basic_iterative',
                                 target=target)
    assert np.min(adv_imgs) >= 0 and np.max(adv_imgs) <= 1

    accuracy = validate_model(sess, x, y, model, adv_imgs, y_benign)
    print('Base accuracy of the target model on adversarial images: ' + str(accuracy))
    print('Generated %d/%d adversarial images' % (int(len(adv_imgs) * (1 - accuracy) + 0.5), len(x_benign)))

    # filter successful attacked adversarial images
    adv_imgs, y_benign, indices = filter_data(sess, x, y, model, adv_imgs, y_benign,
                                              opposite=True)
    x_benign = x_benign[indices]
    sample_num = len(adv_imgs)

    if flip_image:
        probs = model.get_probs(x)
        get_preds = tf.to_float(tf.equal(probs,
                                         tf.reduce_max(probs,
                                                       axis=1, keepdims=True)))
        y_pred = sess.run(get_preds, feed_dict={x: adv_imgs})

        adv_imgs = adv_imgs[:, :, ::-1, :]
        x_benign = x_benign[:, :, ::-1, :]

        flip_direct = validate_model(sess, x, y, model, adv_imgs, y_benign)

        x_flip, y_flip, _ = filter_data(sess, x, y, model, adv_imgs, y_benign, opposite=True,
                                        labels=y_pred)
        backtrack_ratio = 1 - len(x_flip) / float(sample_num)
        flip_accuracy = validate_model(sess, x, y, model, x_flip, y_flip)

        adv_imgs, y_benign, indices = filter_data(sess, x, y, model, adv_imgs, y_benign,
                                                  opposite=False, labels=y_pred)

        x_benign = x_benign[indices]

    if recover_method == 'fg':
        #result_imgs = backtrack_v2(sess, x, model, adv_imgs, recover_params)
        result_imgs = backtrack_v3(sess, x, model, adv_imgs, recover_params)
    elif recover_method == 'fgs':
        recover_params['ord'] = np.inf
        result_imgs = backtracking(sess, x, model, adv_imgs, recover_params)
    else:
        raise NotImplementedError('Recover method should be fg or fgs')

    bt_accuracy = validate_model(sess, x, y, model, result_imgs, y_benign)
    if flip_image:
        print('Preds change rate by flip: ' + str(1 - backtrack_ratio))
        print('Recovery rate (flip only): ' + str(flip_accuracy))
    print('Recovery rate (backtrack): ' + str(bt_accuracy))
    if flip_image:
        acc = (1 - backtrack_ratio) * flip_accuracy + backtrack_ratio * bt_accuracy
        print('Recovery rate (total)    : ' + str(acc))

    print('   %8s%8s%8s%8s%8s%8s' % ('l1', 'l2', 'linf', 'gmax', 'gavg', 'gmin'))
    print('A-B', end=''); eval_perturbation(adv_imgs - x_benign)
    print('R-B', end=''); eval_perturbation(result_imgs - x_benign)
    print('R-A', end=''); eval_perturbation(result_imgs - adv_imgs)

    return x_benign, adv_imgs, result_imgs

if __name__ == '__main__':
    #train_cifar10_classifier('simple', 50)

    # build graph
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model = make_simple_cnn()

    # restore from pretrained
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf_model_load(sess, '../tfmodels/cifar10_simple_model_epoch50')

    # prepare data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    x_filtered, y_filtered, _ = filter_data(sess, x, y, model, x_test, y_test)

    # make sure the model load properly by running it against the test set
    accuracy = validate_model(sess, x, y, model, x_filtered, y_filtered)
    print('Base accuracy of the target model on legitimate images: ' + str(accuracy))

    # initiate attack
    target = None
    attack_method = 'basic_iterative'
    recover_method = 'fg'
    recover_params = {'eps': 8./255,
                      'eps_iter': 1./255,   # only used when ord=np.inf
                      'nb_iter': 100,
                      'step_scale': 0.1,
                      'clip_min': 0.,
                      'clip_max': 1.,
                      'ord': 2}
    benign_imgs, adv_imgs, result_imgs = \
        attack_and_recover(sess, x, y, x_filtered, y_filtered, model, target,
                           attack_method, recover_method, recover_params,
                           flip_image=True)


    #adv_imgs = np.load('data.npy')

    #eval_adv(sess, x, model, adv_imgs, y_filtered)



    # smoothing
    '''
    noisy = tf.placeholder(tf.float32, [None, 32, 32, 3])
    enlarge = tf.image.resize_images(noisy, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    restore = tf.image.resize_images(enlarge, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    adv_imgs = sess.run(restore, feed_dict={noisy: adv_imgs})
    '''

    # recover


    # manual linf clipping
    #epsilon = 8./255
    #linf_clipped = np.clip(result_imgs, adv_imgs - epsilon, adv_imgs + epsilon)
