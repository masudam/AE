from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time



class CAE():
    def __init__(self, learning_rate):
        # Training Parameters
        self.learning_rate = learning_rate

        # Network Parameters
        self.input = [None, 180, 160, 1] # data input (img shape: 180*160*1)
        self.channel1 = 1
        self.channel2 = 32
        self.z_size = 200
        self.fc_size = [None, self.input[1]/4, self.input[2]/4, self.channel2]
        self.fc_prod = int(np.prod(self.fc_size[1:])) # 積

        self.X = tf.placeholder("float", self.input)

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([5, 5, self.channel1, self.channel2])),
            'encoder_h2': tf.Variable(tf.random_normal([5, 5, self.channel2, self.channel2])),
            'encoder_fc' : tf.Variable(tf.random_normal([self.fc_prod, self.z_size])),
            'decoder_fc': tf.Variable(tf.random_normal([self.z_size, self.fc_prod])),
            'decoder_h1': tf.Variable(tf.random_normal([5, 5, self.channel2, self.channel2])),
            'decoder_h2': tf.Variable(tf.random_normal([5, 5, self.channel1, self.channel2])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.channel1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.channel2])),
            'encoder_fc': tf.Variable(tf.random_normal([self.z_size])),
            'decoder_fc': tf.Variable(tf.random_normal([self.fc_prod])),
            'decoder_b1': tf.Variable(tf.random_normal([self.channel2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.channel1])),
        }
        # Construct model
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
        self.loss = self.calc_loss()
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        tf.add_to_collection("X", self.X)
        tf.add_to_collection("encoder", self.encoder_op)
        tf.add_to_collection("en_decoder", self.decoder_op)
        tf.add_to_collection("loss", self.loss)
        tf.add_to_collection("optimizer", self.optimizer)

    # Building the encoder
    def encoder(self,x):
        self.en_1 = tf.nn.relu(tf.nn.conv2d(x, self.weights['encoder_h1'], strides=[1, 2, 2, 1], padding='SAME') + self.biases['encoder_b1']) # 90*80*32
        self.en_2 = tf.nn.relu(tf.nn.conv2d(self.en_1, self.weights['encoder_h2'], strides=[1, 2, 2, 1], padding='SAME') + self.biases['encoder_b2']) # 45*40*32
        self.en_2 = tf.reshape(self.en_2, [-1, self.fc_prod])
        self.en_3 = tf.nn.relu(tf.matmul(self.en_2, self.weights['encoder_fc']) + self.biases['encoder_fc'])
        return self.en_3

    # Building the decoder
    def decoder(self,x):
        layer_1 = tf.nn.relu(tf.matmul(x, self.weights['decoder_fc']) + self.biases['decoder_fc'])
        layer_1 = tf.reshape(layer_1, [-1, 45, 40, 32])
        layer_2 = tf.nn.relu(tf.nn.conv2d_transpose(layer_1, self.weights['decoder_h1'],output_shape=tf.shape(self.en_1),strides=[1, 2, 2, 1],padding='SAME') + self.biases['decoder_b1'])
        layer_3 = tf.nn.sigmoid(tf.nn.conv2d_transpose(layer_2, self.weights['decoder_h2'],output_shape=tf.shape(self.X),strides=[1, 2, 2, 1],padding='SAME') + self.biases['decoder_b2'])
        return layer_3

    def calc_loss(self):
        # Prediction
        y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        y_true = self.X
        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        return loss



# 画像をとってくる
def make_img(sess):
    dir_name="../img_data"
    #fileの数を調べる
    files = os.listdir(dir_name)
    count = len(files)
    imgs = []
    holder = tf.placeholder(tf.string)
    img = tf.read_file(holder)
    img = tf.image.decode_image(img, channels=1)
    img = tf.cast(img,dtype=np.float32)
    img = img/255.0 # 正規化
    for i in range(count):
        img_name = dir_name + "/{}.png".format(str(i).zfill(4))
        img_val = sess.run(img, feed_dict={holder: img_name})
        imgs.append(img_val)
    return np.asarray(imgs, dtype=np.float32)

if __name__ == "__main__":
    # Training Parameters
    num_steps = 3000000
    batch_size = 100
    display_step = 1000
    learning_rate = 0.0001

    with tf.Session() as sess:
        model = CAE(learning_rate = learning_rate)

        # Run the initializer
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        print("prepare data...")
        images = make_img(sess)
        print("done.")

        before = time.time()

        # Training
        for i in range(1, num_steps+1):
            # Prepare Data
            next_b = i % 4 + 1
            batch_x = images[(next_b-1) * batch_size:next_b * batch_size]

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
