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
        saver = tf.train.import_meta_graph(model_dir+".meta")
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)
        # これで前にaddした名前を呼び出して使える
        ph_X = tf.get_collection("X:0")[0]
        decoder = tf.get_collection("en_decoder:0")[0]
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
            g = sess.run(decoder, feed_dict={ph_X: batch_x})

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
