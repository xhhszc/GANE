import tensorflow as tf
import cPickle


class GEN():

    def compute_cosine_diatance_batch(self, x, Y):
        Y_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(Y), axis=1)),[1,-1])
        x_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)),[-1,1])
        xY = tf.matmul(x, Y, transpose_a=False,transpose_b=True)
        d = tf.divide(xY, tf.multiply(Y_norm, x_norm))
        return d



    def __init__(self, autherNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=1e-3, lr_decay_step=200):
        self.autherNum = autherNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.g_params = []

        with tf.variable_scope('generator'):
            if self.param == None:
                self.auther_embeddings = tf.Variable(
                    tf.random_uniform([self.autherNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
            else:
                self.auther_embeddings = tf.Variable(self.param[0])

            self.g_params = [self.auther_embeddings]

        self.auther = tf.placeholder(tf.int32, shape = [None])
        self.co = tf.placeholder(tf.int32, shape = [None])
        self.reward = tf.placeholder(tf.float32, shape = [None])

        self.a_embedding = tf.nn.embedding_lookup(self.auther_embeddings, self.auther)
        self.co_embedding = tf.nn.embedding_lookup(self.auther_embeddings, self.co)
 
        self.all_logits = self.compute_cosine_diatance_batch(self.a_embedding, self.auther_embeddings)
        self.softmax_logits = tf.nn.softmax(self.all_logits)
        co_flattened = tf.range(0, tf.shape(self.co)[0]) * self.autherNum + self.co
        self.i_prob = tf.gather(
            tf.reshape(self.softmax_logits, [-1]),
            co_flattened)
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward)+ self.lamda * (
            tf.nn.l2_loss(self.a_embedding) + tf.nn.l2_loss(self.co_embedding))

        g_opt = tf.train.RMSPropOptimizer(self.learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, self.global_step, var_list=self.g_params)

        self.all_rating = self.softmax_logits

    def save_model(self, sess, filename):
        param = sess.run(self.g_params)
        cPickle.dump(param, open(filename, 'w'))
