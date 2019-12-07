import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import tensorflow as tf
from dis_model import DIS
from gen_model import GEN
import cPickle
import numpy as np
import utils as ut
import multiprocessing

cores = 8 #multiprocessing.cpu_count()

#########################################################################################
# Hyper-parameters
#########################################################################################
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('suffix', '128d', 'some points mark')
flags.DEFINE_integer('emb_dim', 128, 'embedding dimension')
flags.DEFINE_integer('node_num', 10541, 'the node num of network')
flags.DEFINE_integer('batch_size', 256, 'batch size for tarining')
flags.DEFINE_float('init_delta', 0.05, 'init delta for parameter initialization')
flags.DEFINE_integer('epochs', 200, 'the number of training epoch')
flags.DEFINE_float('init_lr_gen', 1e-4, 'init learning rate for generator')
flags.DEFINE_integer('lr_decay_iter_gen', 21082, 'the decay rate for training generator')
flags.DEFINE_float('init_lr_dis', 1e-4, 'init learning rate for discriminator')
flags.DEFINE_integer('lr_decay_iter_dis', 4000, 'the decay rate for training discriminator')

if __name__ == '__main__':
    for arg in FLAGS.__flags:
        the_arg = eval("FLAGS." + arg)
        print(arg, the_arg)

EMB_DIM = FLAGS.emb_dim
AUTHER_NUM = FLAGS.node_num
BATCH_SIZE = FLAGS.batch_size
INIT_DELTA = FLAGS.init_delta

workdir = '../data/'
outputdir = 'output-%s/' % FLAGS.suffix
DIS_TRAIN_FILE = outputdir + 'dis-train.txt'

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
#########################################################################################
# Load data
#########################################################################################
auther_pos_train = {}
with open(workdir + 'train_3.txt')as fin:
    for line in fin:
        line = line.split('\t')
        aid1 = int(line[0])
        aid2 = int(line[1])
        if aid1 in auther_pos_train:
            auther_pos_train[aid1].append(aid2)
        else:
            auther_pos_train[aid1] = [aid2]

        if aid2 in auther_pos_train:
            auther_pos_train[aid2].append(aid1)
        else:
            auther_pos_train[aid2] = [aid1]

auther_pos_test = {}
with open(workdir + 'test_3.txt')as fin:
    for line in fin:
        line = line.split('\t')
        aid1 = int(line[0])
        aid2 = int(line[1])
        if aid1 in auther_pos_test:
            auther_pos_test[aid1].append(aid2)
        else:
            auther_pos_test[aid1] = [aid2]

        if aid2 in auther_pos_test:
            auther_pos_test[aid2].append(aid1)
        else:
            auther_pos_test[aid2] = [aid1]
for key in auther_pos_test:
    auther_pos_test[key] = list(set(auther_pos_test[key]))

#read labels
auther_labels = []
with open(workdir + 'label_3.txt')as fin:
    for line in fin:
        line = line.split('\t')
        label = int(line[1])
        auther_labels.append(label)
auther_labels = np.array(auther_labels)

#########################################################################################
# precision for test
#########################################################################################
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]#converted to a float type, and get top-k
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def recall(r,k):
    num_pos = np.sum(r)
    num_correct = np.sum(r[:k])
    return float(num_correct)/float(num_pos)

def map_metric(r):
    r_len = len(r)
    temp = 0.0
    count = 0.0
    for i in range(r_len):
        if r[i] == 1:
            count += 1
            temp += count/float(i+1)
    return temp/count

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

#get score for the ranking - user
def simple_test_one_user(rating_u):
    rating = rating_u[0]
    u = rating_u[1]
    item_score = []
    for i in range(AUTHER_NUM):
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in auther_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_1 = np.mean(r[:1])
    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    map_m = map_metric(r)
    recall_1 = recall(r, 1)
    recall_3 = recall(r, 3)
    recall_5 = recall(r, 5)
    recall_10 = recall(r, 10)
    recall_15 = recall(r, 15)
    recall_20 = recall(r, 20)

    return np.array([p_1, p_3, p_5, p_10, map_m, recall_1, recall_3, recall_5, recall_10, recall_15, recall_20])

def simple_test_batch_user(batch_rating, batch_user):
    batch_result = []
    for i in range(len(batch_user)):
        result = simple_test_one_user(batch_rating[i], batch_user[i])
        batch_result.append(result)
    return batch_result

#get the model result for test set
def simple_test(sess, model):
    result = np.array([0.] * 11)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = auther_pos_test.keys()
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.auther: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_for_d(sess, model, filename):
    fout = open(filename, 'w')
    for u in auther_pos_train:
        pos = auther_pos_train[u]
        rating = sess.run(model.all_rating, {model.auther: [u]})#rating of all items for u
        prob = np.array(rating[0])

        neg = np.random.choice(np.arange(AUTHER_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            fout.write(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]) + '\n')

    fout.close()


def main():
    i_file_output = 0
    print "load model..."
    generator = GEN(AUTHER_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                    learning_rate=FLAGS.init_lr_gen, lr_decay_step=FLAGS.lr_decay_iter_gen)
    discriminator = DIS(AUTHER_NUM, EMB_DIM, lamda=0.01 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                        learning_rate=FLAGS.init_lr_dis, lr_decay_step=FLAGS.lr_decay_iter_dis)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    dis_log = open(outputdir + 'dis_log.txt', 'w')
    gen_log = open(outputdir + 'gen_log.txt', 'w')

    # minimax training
    best_gen = 0.
    best_dis = 0.
    draw_count_D = 0
    draw_count_G = 0
    for epoch in range(FLAGS.epochs): #5000
        if epoch >= 0:
            # Train D 
            generate_for_d(sess, generator, DIS_TRAIN_FILE)
            train_size = ut.file_len(DIS_TRAIN_FILE)# generate file length
            for d_epoch in range(5):
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_auther, input_coauther_real, input_coauther_fake = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_auther, input_coauther_real, input_coauther_fake = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                train_size - index + 1)
                    index += BATCH_SIZE

                    _ = sess.run([discriminator.d_updates, discriminator.clip_D],
                                 feed_dict={discriminator.auther: input_auther, discriminator.co_real: input_coauther_real,
                                            discriminator.co_fake: input_coauther_fake})
            result = simple_test(sess, discriminator)
            buf = '\t'.join([str(x) for x in result])
            dis_log.write(str(epoch) + '\t' + buf + '\n')
            dis_log.flush()

            p_5 = result[2]
            if p_5 > best_dis:
                print 'best_dis: ', epoch, result
                best_dis = p_5
                discriminator.save_model(sess, outputdir + "gan_discriminator.pkl")
        # Train G
        for g_epoch in range(1):
            for u in auther_pos_train:
                sample_lambda = 0.2
                pos = list(set(auther_pos_train[u]))
                sample_times = 128
                

                rating = sess.run(generator.softmax_logits, {generator.auther: [u]})
                prob = np.reshape(rating, [-1])

                sample = np.random.choice(np.arange(AUTHER_NUM), size=sample_times, p=prob)
                ###########################################################################
                # Get reward and adapt it with importance sampling
                ###########################################################################
                reward = sess.run(discriminator.reward, {discriminator.auther: np.tile(u, (sample_times)), discriminator.co_fake: sample})
                ###########################################################################
                # Update G
                ###########################################################################
                _ = sess.run(generator.gan_updates,
                             {generator.auther: np.tile(u, (sample_times)), generator.co: sample, generator.reward: reward})
        result = simple_test(sess, generator)
        buf = '\t'.join([str(x) for x in result])
        gen_log.write(str(epoch) + '\t' + buf + '\n')
        gen_log.flush()

        p_5 = result[2]
        if p_5 > best_gen:
            print 'best_gen: ', epoch, result
            best_gen = p_5
            generator.save_model(sess, outputdir + "gan_generator.pkl")
            draw_count_G += 1
    gen_log.close()
    dis_log.close()


if __name__ == '__main__':
    main()
