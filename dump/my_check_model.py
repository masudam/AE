from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time
import sys


class AE():
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.002
        self.num_steps = 1000000
        self.batch_size = 100
        self.display_step = 1000

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
    img = tf.reshape(img, [-1])
    img = tf.cast(img,dtype=np.float32)
    img = img/255.0 # 正規化
    for i in range(count):
        img_name = dir_name + "/{}.png".format(str(i).zfill(4))
        img_val = sess.run(img, feed_dict={holder: img_name})
        imgs.append(img_val)
    return np.asarray(imgs, dtype=np.float32)

if __name__ == "__main__":
    inp = sys.argv
    model_dir = "../models/" + inp[1] + '/my-model.ckpt'
    with tf.Session() as sess:
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)
        #まずはmodelを読み込む
        print("restore model...")
        model = AE()
        # saver = tf.train.import_meta_graph(model_dir+".meta")
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)
        print("done.")


        print("prepare data...")
        images = make_img(sess)
        print("done.")
        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
        n = 4
        canvas_orig = np.empty((180 * n, 160 * n))
        canvas_recon = np.empty((180 * n, 160 * n))
        print("prepare check...")
        for i in range(n):
            #batch_x = images[i*100:i*100+1]
            batch_x = images[0:4]
            # Encode and decode the digit image
            g = sess.run(model.decoder_op, feed_dict={model.X: batch_x})

            # Display original images
            for j in range(n):
                # Draw the original digits
                changed = batch_x[j].reshape([180, 160])
                canvas_orig[i * 180:(i + 1) * 180, j * 160:(j + 1) * 160] = changed
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                changed = g[j].reshape([180, 160])
                canvas_recon[i * 180:(i + 1) * 180, j * 160:(j + 1) * 160] = changed

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()
