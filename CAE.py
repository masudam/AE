from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time
import cv2



class CAE():
    def __init__(self, learning_rate):
        # Training Parameters
        self.learning_rate = learning_rate

        # Network Parameters
        self.input = [None, 72, 64, 1] # data input (img shape: 180*160*1)
        self.channel1 = 1
        self.channel2 = 16
        self.channel3 = 32
        self.z_size = 200
        self.fc_size = [None, int(self.input[1]/4), int(self.input[2]/4), self.channel3]
        self.fc_prod = int(np.prod(self.fc_size[1:])) # 積

        self.X = tf.placeholder("float", self.input)

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([5, 5, self.channel1, self.channel2])),
            'encoder_h2': tf.Variable(tf.random_normal([5, 5, self.channel2, self.channel3])),
            'encoder_fc' : tf.Variable(tf.random_normal([self.fc_prod, self.z_size])),
            'decoder_fc': tf.Variable(tf.random_normal([self.z_size, self.fc_prod])),
            'decoder_h1': tf.Variable(tf.random_normal([5, 5, self.channel2, self.channel3])),
            'decoder_h2': tf.Variable(tf.random_normal([5, 5, self.channel1, self.channel2])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.channel2])),
            'encoder_b2': tf.Variable(tf.random_normal([self.channel3])),
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
        self.en_1 = tf.nn.tanh(tf.nn.conv2d(x, self.weights['encoder_h1'], strides=[1, 2, 2, 1], padding='SAME') + self.biases['encoder_b1']) # 36*32*16
        self.en_2 = tf.nn.tanh(tf.nn.conv2d(self.en_1, self.weights['encoder_h2'], strides=[1, 2, 2, 1], padding='SAME') + self.biases['encoder_b2']) # 18*16*32
        self.en_2 = tf.reshape(self.en_2, [-1, self.fc_prod])
        self.en_3 = tf.nn.tanh(tf.matmul(self.en_2, self.weights['encoder_fc']) + self.biases['encoder_fc'])
        return self.en_3

    # Building the decoder
    def decoder(self,x):
        layer_1 = tf.nn.tanh(tf.matmul(x, self.weights['decoder_fc']) + self.biases['decoder_fc'])
        layer_1 = tf.reshape(layer_1, [-1, self.fc_size[1], self.fc_size[2], self.channel3])
        layer_2 = tf.nn.tanh(tf.nn.conv2d_transpose(layer_1, self.weights['decoder_h1'],output_shape=tf.shape(self.en_1),strides=[1, 2, 2, 1],padding='SAME') + self.biases['decoder_b1'])
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
class get_img():
    def __init__(self, sess, rev=False):
        self.sess = sess
        self.dir_name="../img_data"
        #fileの数を調べる
        files = os.listdir(self.dir_name)
        self.count = len(files)
        self.holder = tf.placeholder(tf.string)
        img = tf.read_file(self.holder)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize_images(img, [72,64])
        v = tf.Variable(tf.ones([72,64,1], dtype=tf.float32))
        if rev:
            img = v - img/255.0 # 正規化して反転
        self.img = tf.cast(img,dtype=np.float32)

    def make_img(self):
        imgs = []
        for i in range(self.count):
            img_name = self.dir_name + "/{}.png".format(str(i).zfill(4))
            img_val = self.sess.run(self.img, feed_dict={self.holder: img_name})
            imgs.append(img_val)
        return np.asarray(imgs, dtype=np.float32)

# dirを作る
def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    # Training Parameters
    epoch = 2000000
    batch_size = 32
    display_step = 500
    lr = 0.001
    save_num = 125000

    with tf.Session() as sess:
        model = CAE(learning_rate = lr)
        gm = get_img(sess, rev=True)

        # Run the initializer
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        print("prepare data...")
        images = gm.make_img()
        print("done.")

        dir_path = datetime.datetime.today().strftime("../models/%Y_%m_%d_%H_%M")
        print("model will be saved to {}".format(dir_path))
        my_makedirs(dir_path)

        before = time.time()
        num_data = 400
        # Training
        for i in range(1, epoch+1):
            # Prepare Data
            sff_idx = np.random.permutation(num_data)
            for idx in range(0, num_data, batch_size):
                batch_x = images[sff_idx[idx: idx + batch_size if idx + batch_size < num_data else num_data]]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([model.optimizer, model.loss], feed_dict={model.X: batch_x})
                # Display logs per step
            if i % display_step == 0 or i == 1:
                sec = time.time() - before
                logs = 'Step {}'.format(str(i)) + ' / {} : '.format(str(epoch)) +  'Minibatch Loss: %f' % (l) + " and time is " + str(int(sec))
                print(logs)
                with open (dir_path+"/log.txt",'a') as f:
                    f.write(logs + '\n')
            if i % save_num == 0:
                saver.save(sess, dir_path + '/my-model.ckpt', global_step=int(i/save_num))
