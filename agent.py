
import tensorflow as tf

import copy
import time
import random
import numpy as np
import gcn.inits as gcn_init

import config

class pgAgent:
    def __init__(self, env, nb_warm_up, policy, testPolicy, gamma, lr, memory_limit, batchsize,
                 train_interval):
        np.random.seed(1234567)
        tf.set_random_seed(1234567)

        self.env = env
        self.policy = policy
        self.testPolicy = testPolicy
        self.gamma = gamma
        self.learningRate = lr
        self.memory = []
        self.memory_limit = memory_limit
        self.nb_warm_up = nb_warm_up
        self.batch_size = batchsize
        self.train_interval = train_interval

        self.get_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state("models/")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.ob = None
        self.r = None

        self.episode_count = 0
        self.episode_reward = [.0, .0, .0]

        self.graph = self.get_graph()

    def fit(self, nb_steps):
        observation = self.env.reset()
        for i in range(self.nb_warm_up):
            if i%100==0:
                print("warm up step:", i)
            while True:
                action = self.get_action(observation)
                self.ob, self.r, done, info = self.env.step(action)
                self.save_memory(observation, action, self.r, done, info)
                observation = self.ob
                if done:
                    observation = self.env.reset()
                    break

        self.env.reset()
        time1 = time.time()
        # epi_rewards = open('episodes.txt','w')

        for i in range(nb_steps):
            print("train step:", i)
            while True:
                # print "train step:", i
                action = self.get_action(observation)
                self.ob, self.r, done, info = self.env.step(action)
                self.save_memory(observation, action, self.r, done, info)
                observation = self.ob
                if done and i%self.train_interval==0:
                    # print "before bp time:", (time.time() - time1) / 60
                    for j in range(40):
                        batch_doc, batch_mask, batch_G, batch_action, batch_reward= self.sample_memory(self.batch_size)
                        self.sess.run(self.train_op, feed_dict={
                            self.doc_input: np.array(batch_doc),  # shape=[None, n_obs]
                            self.mask_input: np.array(batch_mask),
                            self.G: np.array(batch_G),
                            self.tf_acts: np.array(batch_action),  # shape=[None, ]
                            self.tf_vt: batch_reward,  # shape=[None, ]
                        })
                    # print "mean reward: ", np.mean(batch_reward)
                if done:
                    # print self.r
                    # epi_rewards.write(str(self.r)+'\n')
                    # epi_rewards.write(str(self.env.new_city_hole.tolist())+'\n')
                    # print "round time:", (time.time() - time1)/60
                    time1 = time.time()
                    observation = self.env.reset()
                    break

    def test(self):
        rs = []
        for i in range(10):
            observation = self.env.reset()
            step = 0
            while True:
                action = self.get_test_action(observation)
                self.ob, self.r, done, info = self.env.step(action)
                observation = self.ob
                step += 1
                if done:
                    # TODO print sth
                    rs.append(self.r)
                    break
        print('test reward',rs, 'mean', np.mean(rs))
        self.test_reward.write(str(rs) + '\n')
        self.test_reward.close()
        self.test_reward = open("test_reward.txt", 'a')
        self.saver.save(self.sess, "models/model.ckpt")

    def get_action(self, observation):
        doc = observation[0]
        mask = observation[1]

        probs = self.sess.run(self.all_act_prob, feed_dict={self.doc_input: doc[np.newaxis, :],
                                                            self.mask_input: mask[np.newaxis, :],
                                                            self.G: self.graph[np.newaxis, :]})
        self.policy.set_mask(mask)
        action = self.policy.select_action(probs[0])
        return action

    def get_test_action(self, observation):
        doc = observation[0]
        mask = observation[1]
        probs = self.sess.run(self.all_act_prob, feed_dict={self.doc_input: doc[np.newaxis, :],
                                                            self.mask_input: mask[np.newaxis, :],
                                                            self.G: self.graph[np.newaxis, :]})
        self.testPolicy.set_mask(mask)
        action = self.testPolicy.select_action(probs[0])
        return action

    def save_memory(self, ob, action, reward, done, info):
        self.memory.append([ob, action, reward, done, info]) # pre_sum and v
        if done:
            self.episode_count += 1

            # TODO only for 3 sentences
            last_reward = self.memory[-1][2]
            self.memory[-1][2] = self.memory[-1][2] - self.memory[-2][2]
            self.memory[-2][2] = self.memory[-2][2] - self.memory[-3][2]
            self.memory[-3][2] = last_reward - self.memory[-1][2] - self.memory[-2][2]

            self.memory[-2][2] += self.gamma * self.memory[-1][2]
            self.memory[-3][2] += self.gamma * self.memory[-2][2]

            self.episode_reward[0] += self.memory[-3][2]
            self.episode_reward[1] += self.memory[-2][2]
            self.episode_reward[2] += self.memory[-1][2]

        if len(self.memory) > self.memory_limit:
            self.episode_count -= 1
            self.episode_reward[0] -= self.memory[0][2]
            self.episode_reward[1] -= self.memory[1][2]
            self.episode_reward[2] -= self.memory[2][2]
            del self.memory[0]
            del self.memory[0]
            del self.memory[0]


    def sample_memory(self, batch_size):
        batch_doc = np.zeros([batch_size, config.doc_max_len, config.sentence_max_len, 200])
        batch_mask = np.zeros([batch_size, config.doc_max_len])
        batch_G = np.zeros([batch_size,config.doc_max_len, config.doc_max_len])
        batch_action = np.zeros((batch_size, ))
        batch_reward = np.zeros((batch_size, ))

        for i in range(batch_size):
            index = random.randint(0, len(self.memory)-1)
            # index = -1
            batch_doc[i] = self.memory[index][0][0]
            batch_mask[i] = self.memory[index][0][1]
            batch_G[i] = self.graph
            batch_action[i] = self.memory[index][1]
            # TODO more careful reward balance
            batch_reward[i] = self.memory[index][2] - self.episode_reward[i%3]/self.episode_count

        return batch_doc, batch_mask, batch_G, batch_action, batch_reward


    def get_net(self):
        with tf.name_scope('inputs'):
            self.doc_input = tf.placeholder(tf.float32,[None, config.doc_max_len, config.sentence_max_len, 200],
                                            name='doc_input')
            self.mask_input = tf.placeholder(tf.float32, [None, config.doc_max_len], name='mask_input')
            self.G = tf.placeholder(tf.float32, [None, config.doc_max_len, config.doc_max_len], name='graph')
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

            self.gcn_attetions = {}
            self.gcn_weights = {}
            self.gcn_bias = {}

        reshaped_doc = tf.reshape(self.doc_input, [-1, config.sentence_max_len, 200])

        conv1 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=1, padding='valid',activation=tf.nn.relu)
        max_pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=config.sentence_max_len, strides=1)

        conv2 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=2, padding='valid',activation=tf.nn.relu)
        max_pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=config.sentence_max_len-1, strides=1)

        conv3 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=3, padding='valid',activation=tf.nn.relu)
        max_pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=config.sentence_max_len-2, strides=1)

        conv4 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=4, padding='valid',activation=tf.nn.relu)
        max_pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=config.sentence_max_len-3, strides=1)

        conv5 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=5, padding='valid',activation=tf.nn.relu)
        max_pool5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=config.sentence_max_len-4, strides=1)

        conv6 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=6, padding='valid',activation=tf.nn.relu)
        max_pool6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=config.sentence_max_len-5, strides=1)

        conv7 = tf.layers.conv1d(inputs=reshaped_doc, filters=50, kernel_size=7, padding='valid',activation=tf.nn.relu)
        max_pool7 = tf.layers.max_pooling1d(inputs=conv7, pool_size=config.sentence_max_len-6, strides=1)

        con = tf.concat([max_pool1, max_pool2, max_pool3, max_pool4, max_pool5, max_pool6, max_pool7], axis=-2)

        new_doc = tf.reshape(con, [-1, config.doc_max_len, 350])

        reshaped_mask = tf.reshape(self.mask_input, [-1, config.doc_max_len, 1])
        new_doc = tf.concat([new_doc, reshaped_mask], axis=-1)

        H1 = self.my_gcn(new_doc, 351, 64, self.G, 0)

        H2 = self.my_gcn(H1, 64, 32, self.G, 1)

        H3 = tf.layers.dense(H2,1)

        all_act = tf.reshape(H3, [-1, config.doc_max_len])

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            reg_loss = tf.reduce_sum(tf.square(all_act))
            pg_loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            loss = pg_loss + 0.0001 * reg_loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learningRate).minimize(loss)


    def my_gcn(self, inputs, input_dim, output_dim, G, index):
        self.gcn_attetions[index] = gcn_init.glorot([config.doc_max_len, config.doc_max_len])
        self.gcn_bias[index] = gcn_init.zeros([config.doc_max_len, output_dim])
        self.gcn_weights[index] = gcn_init.glorot([input_dim, output_dim])

        attention_G = tf.multiply(tf.nn.softmax(self.gcn_attetions[index]), G)

        out = tf.matmul(attention_G, inputs)

        weighted = tf.matmul(tf.reshape(out,[-1,input_dim]), self.gcn_weights[index])

        return tf.reshape(tf.nn.relu(weighted), [-1, config.doc_max_len, output_dim])

    def get_graph(self):
        return np.ones([config.doc_max_len, config.doc_max_len], dtype=np.float32)