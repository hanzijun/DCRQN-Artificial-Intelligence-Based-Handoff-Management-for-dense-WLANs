"""
This part code is the Recurrent Nrural Network + DQN brain, which is a brain of the mobile station.
All decisions are made in here. Using Tensorflow to build the neural network.
Using:
Tensorflow: 1.0
"""""


import numpy as np
import tensorflow as tf
from collections import deque
import pickle
import os
import random
''''
np.random.seed(1)
tf.set_random_seed(1)
'''


class RNNNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            param_file=None
    ):
        self.seqlen = 1
        self.num_layers = 3
        self.n_hidden_units = 10
        self.param_file = param_file
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = self.loadReplayMemory()

        # consist of [target_net, evaluate_net]
        self.sess = tf.Session()
        self.buildRNNNetwork()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        if self.param_file is not None:
            self.saver.restore(self.sess, "./eval_network/rlnew.ckpt")
            print  'loading previous neural network params'

    def buildRNNNetwork(self):
        '''
        Multiple RNN cells
        :param:  RNN networks consists of input layer (l1), multiple RNN cells (RNN cell) and output layer (l2). 
        :param:  collections (c_names) are the collections to store variables.
        :param:  state divided into two parts: the main line  and  sub-line.
        :param: outputs is a list that consists of all the result calculated in every step. 
         [lstm_cell] * self.num_layers represents the number of RNN cells.
        return: bulit RNN networks
        RNN network includes two parts as well, one is eval_network used to learn and update params
        another is target_network which is used to convert "nextstate" to "q_target", the loss of "q_target" 
        and "q_eval" updates the actions_value.  
        The main format is : 
        Q(s,a, t_) = (1-lanmda) * Q(s,a,t) + lambda(reward +gama * max Q(s_,a,t)       
        '''''

        # +++++++++++++++++++build eval_network+++++++++++++++++++++
        self.s = tf.placeholder(tf.float32, [None,  self.seqlen,  self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1 = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.n_hidden_units
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                X1 = tf.reshape(self.s, [-1, self.n_features], name = '2_2D')
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [ n_l1,], initializer=b_initializer, collections=c_names)
                result_in = tf.matmul(X1, w1) + b1
                # result_in = tf.nn.relu(result_in)

                # w3 = tf.get_variable('w3', [self.n_hidden_units,self.n_hidden_units], initializer=w_initializer, collections=c_names)
                # b3 = tf.get_variable('b3', [self.n_hidden_units,],  initializer=b_initializer, collections=c_names)
                # result = tf.nn.relu(tf.matmul(result_in, w3) + b3)
                X_in1 = tf.reshape(result_in, [-1, self.seqlen, self.n_hidden_units], name='2_3D')

            with tf.variable_scope("rnn"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, state_is_tuple = True)
                # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in1, initial_state=None, dtype=tf.float32, time_major=False)
                outputs1 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

            with tf.variable_scope('l2'):
                #out_X = tf.reshape(outputs, [-1, self.n_hidden_units], name='2_2D')
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [ self.n_actions,], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(outputs1[-1], w2) + b2
                # self.q_eval = tf.nn.relu(self.q_eval)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # +++++++++++++++++++++build target_net +++++++++++++++++++++
        self.s_ = tf.placeholder(tf.float32, [None, self.seqlen,  self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                X2 = tf.reshape(self.s_, [-1, self.n_features], name='2_2D')
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [ n_l1,], initializer=b_initializer, collections=c_names)
                result_in2 = tf.matmul(X2, w1) + b1
                # result_in2 = tf.nn.relu(result_in2)

                # w3 = tf.get_variable('w3', [self.n_hidden_units, self.n_hidden_units], initializer=w_initializer,collections=c_names)
                # b3 = tf.get_variable('b3', [self.n_hidden_units, ], initializer=b_initializer, collections=c_names)
                # result2 =  tf.nn.relu(tf.matmul(result_in2, w3) + b3)
                X_in2 = tf.reshape(result_in2, [-1, self.seqlen, self.n_hidden_units], name='2_3D')

            with tf.variable_scope("rnn"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units,state_is_tuple = True)
                # lstm cell is divided into two parts (c_state, h_state)
                init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in2, initial_state=None, dtype=tf.float32, time_major=False)
                outputs2 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

            with tf.variable_scope('l2'):
                #out_X = tf.reshape(outputs, [-1, self.n_hidden_units], name='2_2D')
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [self.n_actions,], initializer=b_initializer, collections=c_names)
                #self.q_next = tf.matmul(out_X, w2) + b2
                self.q_next = tf.matmul(outputs2[-1], w2) + b2
                # self.q_next = tf.nn.relu(self.q_next)

    def store_transition(self, s, a, r, s_):
        """
        store the current memories in order to restore.
        :param s:  state
        :param a:  action
        :param r:  reward
        :param s_:  nextstate
        :return:  saved memories
        """
        self.memory.append((s,a,r,s_))
        if len(self.memory)> self.memory_size:
            self.memory.popleft()

    def choose_action(self, observation):
        ''''
         to have batch dimension when feed into tf placeholder
        choose actionID to adaptive various environments
        :param observation:  current state
        :return:  action ID
        '''
        observation = observation[np.newaxis, :]
        observation = observation.reshape([1, self.seqlen, self.n_features])
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        '''
        noteworthy: inputs should be converted  from "2D" to "3D" which  RNN needs
        :return:  updated params in DRQN neural networks
        '''
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory

        batch_memory = random.sample(self.memory,self.batch_size)

        nextstateList = np.squeeze([data[3] for data in batch_memory])

        stateList = np.squeeze([data[0] for data in batch_memory])

        nextstateList = nextstateList.reshape([self.batch_size, self.seqlen, self.n_features])

        stateList = stateList.reshape([self.batch_size, self.seqlen, self.n_features])
        # s_list = [i.reshape([1,1,2]) for i in batch_memory[:, -self.n_features:]]
        # slist = [i.reshape([1,1,2]) for i in batch_memory[:, :self.n_features:]]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # self.s_: batch_memory[:, -self.n_features:],  # fixed params
                # self.s: batch_memory[:, :self.n_features]  # newest params
                self.s_: nextstateList,
                self.s: stateList
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        eval_act_index = np.squeeze([data[1] for data in batch_memory]).astype(int)
        reward = np.squeeze([data[2] for data in batch_memory])

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: stateList,
                                             self.q_target: q_target,})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def saveReplayMemory(self):
        print 'Memory Size: ' + str(len(self.memory))
        with open('./eval_network/replayMemory.pkl', 'wb') as handle:
            pickle.dump(self.memory, handle, -1)  # Using the highest protocol available
        pass

    def loadReplayMemory(self):
        if os.path.exists('./eval_network/replayMemory.pkl'):
            with open('./eval_network/replayMemory.pkl', 'rb') as handle:
                replayMemory = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        else:
            replayMemory = deque()
        return replayMemory
