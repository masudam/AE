from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time



class AE():
    def __init__(self, learning_rate):
        # Training Parameters
        self.learning_rate = learning_rate

        # Network Parameters
        self.num_hidden_1 = 1024 # 1st layer num features
        self.num_hidden_2 = 512 # 2nd layer num features (the latent dim)
        self.num_hidden_3 = 128
        self.num_input = 28800 # MNIST data input (img shape: 180*160)

        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [None, self.num_input])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_3])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_3, self.num_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.num_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.num_input])),
        }
        # Construct model
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
        self.loss = self.calc_loss()
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def calc_loss(self):
        # Prediction
        y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        y_true = self.X
        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        return loss


    # Building the encoder
    def encoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        # Encoder Hidden layer with sigmoid activation #3
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']),
                                       self.biases['encoder_b3']))
        return layer_3


    # Building the decoder
    def decoder(self,x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        # Decoder Hidden layer with sigmoid activation #3
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['decoder_h3']),
                                       self.biases['decoder_b3']))
        return layer_3

# 画像をとってくる
def make_img(sess):
    dir_name="../img_data"
    #fileの数を調べる
    files = os.listdir(dir_name)
    count = len(files)
    imgs = []
    for i in range(count):
        img_name = dir_name + "/{}.png".format(str(i).zfill(4))
        img = tf.read_file(img_name)
        img = tf.image.decode_image(img, channels=1)
        img = tf.reshape(img, [-1])
        img = tf.cast(img,dtype=np.float32)
        img = img/255.0 # 正規化
        img_val = sess.run(img)
        imgs.append(img_val)
    return np.asarray(imgs, dtype=np.float32)

# Start Training
# Start a new TF session

# Training Parameters
num_steps = 1000000
batch_size = 100
display_step = 1000

with tf.Session() as sess:
    model = AE(learning_rate = 0.002)

    # Run the initializer
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=100)
    images = make_img(sess)

    before = time.time()

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        #ここにバッチサイズのデータを渡す
        # batch_x, _ = mnist.train.next_batch(batch_size)
        next_b = i % 4 + 1
        batch_x = images[(next_b-1)*100:next_b*100]

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([model.optimizer, model.loss], feed_dict={model.X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            sec = time.time() - before
            print('Step %i: Minibatch Loss: %f' % (i, l) + " and time is " + str(int(sec)))


    def my_makedirs(path):
        if not os.path.isdir(path):
            os.makedirs(path)
    dir_path = datetime.datetime.today().strftime("../models/%Y_%m_%d_%H_%M")
    my_makedirs(dir_path)
    saver.save(sess, dir_path + '/my-model.ckpt')
