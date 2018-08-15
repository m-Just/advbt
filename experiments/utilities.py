from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import numpy as np
import tensorflow as tf

from cleverhans.utils_tf import model_eval, model_loss
from cleverhans.utils import set_log_level
import scipy as sc

def conservative_smoothing(imgs, w = 3):
    b,x,y,c = imgs.shape
    e = int((w - 1)/2)
    new_imgs = np.array(imgs)
    for i in range(b):
        for j in range(e,x-e):
            for k in range(e,y-e):
                for z in range(c):
                    t = new_imgs[i,j,k,z]
                    tmp = imgs[i, j-e:j+e+1, k-e:k+e+1, z].flatten()
                    tmp = np.sort(tmp)
                    if t == tmp[-1]:
                        t = tmp[-2]
                    elif t == tmp[0]:
                        t = tmp[1]
                    new_imgs[i,j,k,z] = t
    return new_imgs

def data_cifar10():
    import keras
    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

def validate_model(sess, x, y, model):
    (x_train, y_train), (x_test, y_test) =data_cifar10()
    # Make sure the model load properly by running it against the test set
    predictions = model.get_probs(x)
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, predictions, X_test=x_test, Y_test=y_test, args=eval_params)
    print('base accuracy of the target model on normal images: ' + str(accuracy))

def predict(sess, x, model, x_test, batch_size = 128):
    pred = model.get_probs(x)
    probs = []
    num_batch = x_test.shape[0]// batch_size
    for i in range(num_batch):
        probs.append(sess.run(pred,feed_dict = {
            x:x_test[i*batch_size:(i+1)*batch_size]
        }
                                )
                       )
    if x_test.shape[0]% batch_size != 0:
        probs.append(sess.run(
                    pred,feed_dict = {x:x_test[num_batch*batch_size:]
                                      }
                )
                               )
    probs = np.concatenate(probs, axis = 0)
    return probs
        
def backtracking(sess, x, y, model,  x_test, y_test, params, batch_size = 128):
    tf.set_random_seed(1822)
    set_log_level(logging.DEBUG)
    from cleverhans.attacks import BasicIterativeMethod
    method = BasicIterativeMethod(model, sess = sess)
    adv_x = method.generate(x, **params)
    num_batch = x_test.shape[0]// batch_size
    adv_imgs = []
    for i in range(num_batch):
        adv_imgs.append(sess.run(adv_x,feed_dict = {
            x:x_test[i*batch_size:(i+1)*batch_size],
            y:y_test[i*batch_size:(i+1)*batch_size]
        }))
    if x_test.shape[0]% batch_size != 0:    
        adv_imgs.append(sess.run(adv_x,feed_dict = {
                x:x_test[num_batch*batch_size:],
                y:y_test[num_batch*batch_size:]
            }))
                    
    adv_imgs = np.concatenate(adv_imgs, axis = 0)
        
    return adv_imgs

def backtracking_v2(sess, x, y, model,  x_test, y_test, params, ftr, ftr_params, batch_size = 128):
    nb_iter = params["nb_iter"]
    eps = params["eps"]
    params["nb_iter"] = 1
    params["eps"] = np.inf
    adv_imgs = x_test
    for _ in range(nb_iter):
        adv_imgs = backtracking(sess,x,y,model, adv_imgs, y_test, params, batch_size)
        adv_imgs = np.clip(adv_imgs, a_min = adv_imgs-eps, a_max = adv_imgs + eps)
        adv_imgs = ftr(adv_imgs,**ftr_params)
    return adv_imgs    

def backtracking_v3(sess, x, model, adv_imgs, y_probs, params, ftr = lambda t:t, ftr_params = {}):
    logits = model.get_logits(x)
    labels = tf.stop_gradient(tf.constant(y_probs))
   
    smoothness = tf.image.total_variation(x)
    
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    grad, = tf.gradients(loss, x)
    normalized_grad = tf.sign(grad)
    adv_x = x + params['eps_iter'] * normalized_grad
    
    #smoothness = tf.expand_dims(tf.image.total_variation(x),axis = -1)
    #sm_grad, = tf.gradients(smoothness, adv_x)
    #sm_normalized_grad = tf.sign(sm_grad)
    #adv_x = adv_x + params['sm_iter'] * sm_normalized_grad
    
    
    
    clip_base = tf.constant(adv_imgs)
    adv_x = tf.clip_by_value(adv_x, clip_base - params['eps'], clip_base + params['eps'])
    adv_x = tf.clip_by_value(adv_x, params['clip_min'], params['clip_max'])

    start_imgs = adv_imgs
    sm_vals = []
    for i in range(params['nb_iter']):
        adv_imgs, loss_val, grad_val, logits_val, sm_val = sess.run([adv_x, loss, grad, logits, smoothness], feed_dict={x: adv_imgs})
        adv_imgs = ftr(adv_imgs, **ftr_params)
        sm_vals.append(sm_val)
        print('Recovery iteration %d: ' % (i + 1))
        print('  loss(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(loss_val), np.min(loss_val), np.max(loss_val)))
        grad_val = np.abs(grad_val)
        print('  grad(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(grad_val), np.min(grad_val), np.max(grad_val)))
        print('  sm(avg)=%f, (min)=%f, (max)=%f' % \
            (np.mean(sm_val), np.min(sm_val), np.max(sm_val)))
    return adv_imgs, sm_vals

def attack_classifier(sess, x, y, model, x_test, y_test, attack_method = "fgsm", target = None, batch_size =128):
    tf.set_random_seed(1822)
    set_log_level(logging.DEBUG)
    
    # Initialize attack
    if attack_method == "fgsm":
        from cleverhans.attacks import FastGradientMethod
        params = {'eps': 8/255,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = FastGradientMethod(model, sess=sess)
        
    elif attack_method == "bim":
        from cleverhans.attacks import BasicIterativeMethod
        params = {'eps':8/255,
                  'eps_iter':1/255,
                  'nb_iter': 10,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        if target is not None:
            params["y_target"] = tf.constant(np.repeat(np.eye(10)[target:target+1], batch_size, axis = 0))
        method = BasicIterativeMethod(model,sess = sess)
        
    elif attack_method == "jsma":
        from cleverhans.attacks import SaliencyMapMethod
        params = {'theta':8/255,
                  'gamma':0.1,
                  'clip_min': 0.,
                  'clip_max': 1.
                 }
        assert target is None
        method = SaliencyMapMethod(model,sess = sess)
        
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
        
    else:
        raise ValueError("Can not recognize this attack method")
    
    
    adv_x = method.generate(x, **params)
    num_batch = x_test.shape[0]// batch_size
    adv_imgs = []
    for i in range(num_batch):
        adv_imgs.append(sess.run(adv_x,feed_dict = {
            x:x_test[i*batch_size:(i+1)*batch_size],
            y:y_test[i*batch_size:(i+1)*batch_size]
        }
                                )
                       )
    if x_test.shape[0]% batch_size != 0:
        adv_imgs.append(sess.run(adv_x,feed_dict = {
                x:x_test[num_batch*batch_size:],
                y:y_test[num_batch*batch_size:]
            }))
        
    adv_imgs = np.concatenate(adv_imgs, axis = 0)
        
    return adv_imgs

def filter_data(sess, x, y, model, x_test, y_test, target = None, batch_size = 128, eval_size =  1280):
    predictions = model.get_probs(x)
    eval_par = {'batch_size': batch_size}
    
    # only take the ones that are correctly classified by the target model
    x_filtered = []
    y_filtered = []
    counter = 0
    for each in zip(x_test,y_test):
        if np.argmax(each[1]) != target and model_eval(sess, x, y, predictions,np.array([each[0]]),np.array([each[1]]),args = {'batch_size':1}) == 1.0:
            x_filtered.append([each[0]])
            y_filtered.append([each[1]])
            counter += 1
        if counter >= eval_size:
            break
    return np.concatenate(x_filtered,axis = 0), np.concatenate(y_filtered, axis = 0)
    
def model_eval_adv(sess, x, y, model, x_eval, y_eval, target = None, batch_size = 128):
    preds_adv =  model.get_probs(x)
    eval_par = {'batch_size': batch_size}
    if target is None:
        acc = model_eval(sess, x, y, preds_adv, x_eval, y_eval,args = eval_par)
        print('Success rate of non target attacks: %0.7f\n' % (1 - acc))
    else:
        acc = model_eval(sess, x, y, preds_adv, x_eval, np.repeat(np.eye(10)[target:target+1], x_eval.shape[0] , axis = 0),args = eval_par)
        print('Success rate of target attacks : %0.7f\n' % acc)

def train_classifier(model_name, nb_epochs):
    tf.set_random_seed(1822)
    set_log_level(logging.DEBUG)

    # Get CIFAR-10 data
    train_start = 0
    train_end = 50000
    test_start = 0
    test_end = 10000
    (x_train, y_train), (x_test, y_test) = \
        data_cifar10(train_start, train_end, test_start, test_end)

    #label_smooth = .1
    #y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Initialize model
    epoch_step = tf.Variable(0, trainable=False)
    if model_name == 'simple':
        model = make_simple_cnn()
        learning_rate = tf.constant(0.001)
    elif model_name == 'resnet':
        model = make_resnet(depth=32)
        learning_rate = tf.constant(0.001)
        '''
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step=epoch_step,
            decay_steps=nb_epochs, 
            decay_rate=0.9441,
            staircase=True)
        learning_rate = tf.case([
            (epoch_step > 180, lambda: 0.5e-6),
            (epoch_step > 160, lambda: 1e-6),
            (epoch_step > 120, lambda: 1e-5),
            (epoch_step > 80, lambda: 1e-4)],
            default=lambda: 1e-3)
        learning_rate = tf.case([
            (epoch_step > 70, lambda: 1e-5),
            (epoch_step > 50, lambda: 1e-4)],
            default=lambda: 1e-3)
        '''
        lr1 = tf.train.polynomial_decay(
            learning_rate,
            global_step=epoch_step,
            decay_steps=80,
            end_learning_rate=1e-4,
            power=0.5)
        lr2 = tf.train.exponential_decay(
            1e-4,
            global_step=epoch_step-80,
            decay_steps=1,
            decay_rate=0.944061)
        learning_rate = tf.cond(epoch_step < 80,
                                true_fn=lambda: lr1,
                                false_fn=lambda: lr2)
    else:
        raise ValueError()
    assert len(model.get_params()) == len(tf.trainable_variables())
    #for layer in model.layers:
    #    print(layer.name)
    preds = model.get_probs(x)

    batch_size = 32
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epoch_step': epoch_step, # used for lr decay
        'weight_decay': 1e-4
    }
    rng = np.random.RandomState([2018, 6, 9])

    sess = tf.Session()

    def evaluate():
        eval_params = {'batch_size': 128}
        acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
        assert x_test.shape[0] == test_end - test_start, x_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    model_train(sess, x, y, preds, x_train, y_train,
                evaluate=evaluate, args=train_params, rng=rng,
                var_list=model.get_params())
    a = sess.run(tf.trainable_variables()[-5])
    print(a)

    savedir = './tfmodels'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    model_savename = 'cifar10_%s_model_epoch%d' % (model_name, nb_epochs)
    saver.save(sess, os.path.join(savedir, model_savename))
