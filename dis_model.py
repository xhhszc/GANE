import tensorflow as tf
import cPickle


class DIS():
    def compute_cosine_distance(self, x, y):
        with tf.name_scope('cosine_distance'):
            x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
            y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
            xy=tf.reduce_sum(tf.multiply(x,y), axis=1)
            d = tf.divide(xy,tf.multiply(x_norm, y_norm))
            return d
    def compute_cosine_diatance_batch(self, x, Y):
        Y_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(Y), axis=1)),[1,-1])
        x_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)),[-1,1])
        xY = tf.matmul(x, Y, transpose_a=False,transpose_b=True)
        d = tf.divide(xY, tf.multiply(Y_norm, x_norm))
        return d


    def __init__(self, autherNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=1e-3, lr_decay_step=100):
        self.autherNum = autherNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate #tf.train.exponential_decay(learning_rate, self.global_step, lr_decay_step, 0.8, staircase=True)
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.auther_embeddings = tf.Variable(
                    tf.random_uniform([self.autherNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
            else:
                self.auther_embeddings = tf.Variable(self.param[0])

        self.d_params = [self.auther_embeddings]

        # placeholder definition
        self.auther = tf.placeholder(tf.int32, shape=[None])
        self.co_real = tf.placeholder(tf.int32, shape=[None])
        self.co_fake = tf.placeholder(tf.int32, shape=[None])

        self.a_embedding = tf.nn.embedding_lookup(self.auther_embeddings, self.auther)
        self.co_real_embedding = tf.nn.embedding_lookup(self.auther_embeddings, self.co_real)
        self.co_fake_embedding = tf.nn.embedding_lookup(self.auther_embeddings, self.co_fake)
        

        self.pre_logits_real = self.compute_cosine_distance(self.a_embedding, self.co_real_embedding) 
        self.pre_logits_fake = self.compute_cosine_distance(self.a_embedding, self.co_fake_embedding) 
        self.pre_loss = -(tf.reduce_mean(self.pre_logits_real) - tf.reduce_mean(self.pre_logits_fake))+ self.lamda * (
            tf.nn.l2_loss(self.a_embedding) + tf.nn.l2_loss(self.co_real_embedding) + tf.nn.l2_loss(self.co_fake_embedding)
        )


        d_opt = tf.train.RMSPropOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, self.global_step, var_list=self.d_params)

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_params]

        #reward for G(fake)
        self.reward = self.compute_cosine_distance(self.a_embedding, self.co_fake_embedding)

        # for test stage, self.u: [batch_size],
        self.all_rating_init = self.compute_cosine_diatance_batch(self.a_embedding, self.auther_embeddings)
        self.all_rating = tf.nn.softmax(self.all_rating_init)

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
