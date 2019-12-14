import tensorflow as tf
import numpy as np



class AE():
    def __init__(self,img_shape,d):
        self.sess = tf.get_default_session
        self.input_shape = img_shape[0]*img_shape[1]
        with tf.variable_scope("AE"):
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_shape))
            self.W1 = tf.Variable(tf.random_normal(shape=(self.input_shape,d)))
            self.b1 = tf.Variable(np.zeros(d).astype(np.float32))
            self.Z = tf.nn.sigmoid(tf.matmul(self.X, self.W1) + self.b1)
            self.W2 = tf.Variable(tf.random_normal(shape=(d,self.input_shape)))
            self.b2 = tf.Variable(np.zeros(self.input_shape).astype(np.float32))
            logits = tf.matmul(self.Z, self.W2) + self.b2
            self.X_hat = tf.nn.sigmoid(logits)
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X,logits=logits))

    def encode(self, X):
        return self.sess().run(self.Z, feed_dict={self.X: X})
    def decode(self, Z):
        return self.sess().run(self.X_hat, feed_dict={self.Z: Z})




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--isRender", help="optional", action="store_true")
    args = parser.parse_args()

    # まずはこの辺で画像をimreadして持ってくる

    #my_ae = AE()
    train = tf.train.AdamOptimizer(learning_rate).minimize(my_ae.loss)




