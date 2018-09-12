# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:58:39 2018

@author: HR
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, reshape=False)
gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
xavier_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=55)
f_dim = 20

def cae(x_in, is_train):
    conv_size, conv_stride, padding = (5, 5), (2, 2), 'same'
    use_bias = False
    reuse = tf.AUTO_REUSE
    with tf.variable_scope('cae', reuse=reuse):
        h1_conv = tf.layers.conv2d(inputs=x_in, filters=f_dim, kernel_size=conv_size, strides=conv_stride, kernel_initializer=xavier_init,
                                   use_bias=use_bias, padding=padding, name='h1/conv', reuse=reuse)
        h1_bn = tf.layers.batch_normalization(inputs=h1_conv, gamma_initializer=gamma_init, training=is_train, name='h1/bn', reuse=reuse)
        h1_act = tf.nn.relu(h1_conv, name='h1/act')
        
        h2_conv = tf.layers.conv2d(inputs=h1_act, filters=f_dim*2, kernel_size=conv_size, strides=conv_stride, kernel_initializer=xavier_init,
                                   use_bias=use_bias, padding=padding, name='h2/conv', reuse=reuse)
        h2_bn = tf.layers.batch_normalization(inputs=h2_conv,  gamma_initializer=gamma_init, training=is_train, name='h2/bn', reuse=reuse)
        h2_act = tf.nn.relu(h2_conv, name='h2/act')
        
        h3_conv = tf.layers.conv2d(inputs=h2_act, filters=f_dim*4, kernel_size=conv_size, strides=(1, 1), kernel_initializer=xavier_init,
                                   use_bias=use_bias, padding=padding, name='h3/conv', reuse=reuse)
        h3_bn = tf.layers.batch_normalization(inputs=h3_conv, gamma_initializer=gamma_init, training=is_train, name='h3/bn', reuse=reuse)
        h3_act = tf.nn.relu(h3_conv, name='h3/act')
        
        h4_conv = tf.layers.conv2d(inputs=h3_act, filters=5, kernel_size=conv_size, kernel_initializer=xavier_init, use_bias=use_bias, padding=padding, name='h4/conv', reuse=reuse)
        h4_bn = tf.layers.batch_normalization(inputs=h4_conv, gamma_initializer=gamma_init, training=is_train, name='h4/bn', reuse=reuse)
        h4_act = tf.nn.relu(h4_conv, name='h4/bn')
        
        h5_convtr = tf.layers.conv2d_transpose(inputs=h4_act, filters=f_dim*4, kernel_size=conv_size, kernel_initializer=xavier_init, use_bias=use_bias, padding=padding, name='h5/convtr')
        h5_bn = tf.layers.batch_normalization(inputs=h5_convtr, gamma_initializer=gamma_init, training=is_train, name='h5/bn', reuse=reuse)
        h5_act = tf.nn.relu(h5_convtr, name='h5/bn')
        
        h6_convtr = tf.layers.conv2d_transpose(inputs=h5_act, filters=f_dim*2, kernel_size=conv_size, strides=(1, 1), kernel_initializer=xavier_init,
                                               use_bias=use_bias, padding=padding, name='h6/convtr', reuse=reuse)
        h6_bn = tf.layers.batch_normalization(inputs=h6_convtr, gamma_initializer=gamma_init, training=is_train, name='h6/bn', reuse=reuse)
        h6_act = tf.nn.relu(h6_convtr, name='h6/bn')
        
        h7_convtr = tf.layers.conv2d_transpose(inputs=h6_act, filters=f_dim, kernel_size=conv_size, strides=conv_stride, kernel_initializer=xavier_init, 
                                               use_bias=use_bias, padding=padding, name='h7/convtr', reuse=reuse)
        h7_bn = tf.layers.batch_normalization(inputs=h7_convtr, gamma_initializer=gamma_init, training=is_train, name='h7/bn', reuse=reuse)
        h7_act = tf.nn.relu(h7_convtr)
        
        output = tf.layers.conv2d_transpose(inputs=h7_act, filters=1, kernel_size=conv_size, strides=conv_stride, kernel_initializer=xavier_init,
                                            use_bias=use_bias, padding=padding, name='output/convtr', reuse=reuse)

    return output

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

max_epoch = 250
mb_size = 50

roc_auc_scores = []
for normal_number in range(10):
    train_dat = mnist.train.images[mnist.train.labels == normal_number]
    n_train = len(train_dat)
    train_batch = chunks(np.arange(n_train), mb_size)
    
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    x_out_train = cae(x_input, is_train=True)
    x_out_test = cae(x_input, is_train=False)
    
    train_recon_sqdiff = tf.squared_difference(x_input, x_out_train)
    train_recon_loss = tf.reduce_sum(train_recon_sqdiff, axis=[1, 2, 3])
    train_cae_loss = tf.reduce_mean(train_recon_loss)
    
    test_recon_sqdiff = tf.squared_difference(x_input, x_out_test)
    test_recon_loss = tf.reduce_sum(test_recon_sqdiff, axis=[1, 2, 3])
        
    cae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cae')
    cae_opt = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9).minimize(loss=train_cae_loss, var_list=cae_vars)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())    

    for epoch in range(max_epoch):
        train_idx = np.arange(n_train)
        np.random.shuffle(train_idx)
        train_batch = chunks(train_idx, mb_size)
        
        batch_loss_stack = []
        for batch_idx in train_batch:
            batch_dat = train_dat[batch_idx]
            batch_loss, _ = sess.run([train_recon_loss, cae_opt], feed_dict={x_input: batch_dat})
            batch_loss_stack += batch_loss.tolist()
            
        print('Epoch [{:3d}/{:3d}] reconstruction loss: {:.5f}'.format(epoch+1, max_epoch, np.mean(batch_loss_stack)))
            
    train_scores = []
    train_batch = chunks(np.arange(n_train), 1000)
    for batch_idx in train_batch:
        batch_dat = train_dat[batch_idx]
        recon_errors = sess.run(test_recon_loss, feed_dict={x_input: batch_dat})
        train_scores += recon_errors.tolist()
        
    threshold = np.quantile(np.array(train_scores), 0.90)
    
    test_normal = mnist.test.images[mnist.test.labels == normal_number]
    test_abnormal = mnist.test.images[mnist.test.labels != normal_number]
    
    normal_scores, abnormal_scores = [], []
    test_idx = np.arange(len(test_normal))
    test_batch = chunks(test_idx, 1000)
    for batch_idx in test_batch:
        batch_dat = test_normal[batch_idx]
        recon_errors = sess.run(test_recon_loss, feed_dict={x_input: batch_dat})
        normal_scores += recon_errors.tolist()
        
    test_idx = np.arange(len(test_abnormal))
    test_batch = chunks(test_idx, 1000)
    for batch_idx in test_batch:
        batch_dat = test_abnormal[batch_idx]
        recon_errors = sess.run(test_recon_loss, feed_dict={x_input: batch_dat})
        abnormal_scores += recon_errors.tolist()
        
    print(np.mean(normal_scores), np.mean(abnormal_scores))
    test_scores = np.concatenate([normal_scores, abnormal_scores])
    test_labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_abnormal))])
    test_predicts = (test_scores >= threshold).astype(np.int)
    
    roc_auc_scores.append(roc_auc_score(test_labels, test_scores))
    plt.title('[ Normal Number - {} | AUC = {:.4f} ]'.format(normal_number, roc_auc_scores[-1]))
    plt.plot(test_scores)
    plt.show()
    print(confusion_matrix(test_labels, test_predicts))
    
    sess.close()
    tf.reset_default_graph()
    
print(roc_auc_scores)