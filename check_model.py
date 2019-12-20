from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time
import sys
from CAE import CAE

# 画像をとってくる
def make_img_for_AE(sess):
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


# 画像をとってくる
class get_img():
    def __init__(self, sess, rev=False):
        self.sess = sess
        self.dir_name="../img_data"
        #fileの数を調べる
        files = os.listdir(self.dir_name)
        self.count = len(files)
        self.holder = tf.placeholder(tf.string, name='holder')
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

if __name__ == "__main__":
    inp = sys.argv
    model_dir = "../models/" + inp[1] + '/my-model.ckpt' + inp[2]

    with tf.Session() as sess:
        gm = get_img(sess, rev=True)# 色反転の時はTrue
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)
        #まずはmodelを読み込む
        print("restore model...")
        saver = tf.train.import_meta_graph(model_dir+".meta")
        #saver = tf.train.Saver()
        saver.restore(sess, model_dir)
        # これで前にaddした名前を呼び出して使える
        ph_X = tf.get_collection("X")[0]
        print(ph_X)
        decoder = tf.get_collection("en_decoder")[0]
        print(decoder)
        print("done.")

        print("prepare data...")
        if inp[3] == "cae":
            images = gm.make_img()
        else:
            images = make_img_for_AE(sess)
        print("done.")

        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
        n = 4
        canvas_orig = np.empty((72 * n, 64 * n))
        canvas_recon = np.empty((72 * n, 64 * n))
        print("prepare check...")
        for i in range(n):
            batch_x = images[i*100:i*100+4]
            #batch_x = images[0:4]
            # Encode and decode the digit image
            g = sess.run(decoder, feed_dict={ph_X: batch_x})

            # Display original images
            loss = tf.reduce_mean(tf.pow(g - batch_x, 2))
            print(sess.run(loss))
            print(loss)
            for j in range(n):
                # Draw the original digits
                changed = batch_x[j].reshape([72, 64])
                canvas_orig[i * 72:(i + 1) * 72, j * 64:(j + 1) * 64] = changed
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                changed = g[j].reshape([72, 64])
                canvas_recon[i * 72:(i + 1) * 72, j * 64:(j + 1) * 64] = changed

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()
