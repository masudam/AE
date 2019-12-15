from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time


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



if __name__ == "__main__":
    inp = sys.argv
    model_dir = "../" + inp[1] + '/my-model.ckpt'
    with tf.Session() as sess:
        #まずはmodelを読み込む
        saver = tf.train.import_meta_graph(model_dir+".meta")
        saver.restore(sess, model_dir)

        images = make_img(sess)
        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
        n = 4
        canvas_orig = np.empty((180 * n, 160 * n))
        canvas_recon = np.empty((180 * n, 160 * n))
        for i in range(n):
            batch_x = images[:n]
            # Encode and decode the digit image
            g = sess.run(decoder_op, feed_dict={X: batch_x})

            # Display original images
            for j in range(n):
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    g[j].reshape([28, 28])

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()
