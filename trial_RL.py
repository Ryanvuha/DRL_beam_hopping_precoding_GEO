# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:19:39 2021

@author: mnguy
## 
"""


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pylab as plt
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import scipy.io as sio
from parameters import parameters, write_para
import pickle







REPLAY_MEMORY_SIZE = 10_000  # How many last steps to keep for model training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)

MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

LEARNING_RATE = 0.1
DISCOUNT = 0.99
# Environment settings
#EPISODES = 20_000
EPISODES = 10000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# # initialize
# write_para()
file_Name = "parameters_data" # depends on the file name that we want to save
fileObject = open(file_Name,'rb')  
para = pickle.load(fileObject)

num_bits = np.log2(para.num_codework) + para.num_per_recent*para.num_recent + para.num_indicateDRLorDNN
num_bits_int = np.ceil(num_bits)


num_actions = para.num_actions
num_states = para.num_states


try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


class DQNAgent:
    def __init__(self):
        self._num_states = num_states # input of DRL
        self._num_actions = num_actions # num_actions = para.Ntotal_urllc*2^num_cluster, output of DRL
        
        self._batch_size = MINIBATCH_SIZE
        self.num_per_recent = para.num_per_recent
        self.num_recent = para.num_recent 
        
        
        self.model = self.define_model()
        self.target_model = self.define_model()
        self.target_model.set_weights(self.model.get_weights()) # two networks with the same initial weights 
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #replay1#
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def define_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim= self._num_states, activation='relu'))
        model.add(Dense(50, activation='relu'))
        #model.add(Dense(50, activation='relu'))
        model.add(Dense(self._num_actions, activation='relu'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
        
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
            
    
    
    def convert_array_action(self, action):
        # action is integer number
        yy = bin(int(action))[2:]
        xx = yy[::-1]
        
        
        action_indicate = np.array(int(xx[0]))
        action_recent = np.zeros((self.num_recent,1))
        for kk in range(self.num_recent):
            a2 =  [int(x) for x in xx[kk*self.num_per_recent+1:(kk+1)*self.num_per_recent+1]]
            a3 = np.asarray(a2)
            ss = 0
            for jj in range(a3.shape[0]):
                ss = ss + 2**jj*a3[jj]
                
            action_recent[kk] = int(2**self.num_per_recent*kk + ss)
            
        a2 = [int(x) for x in xx[(self.num_recent)*self.num_per_recent+1:]] 
        a3 = np.asarray(a2)
        ss = 0
        for jj in range(len(a3)):
            ss = ss + 2**jj*a3[jj]
        action_refine = ss
            
        #action_out = np.concatenate((a1,a4,a3))
            
        return action_indicate, action_recent, action_refine
        
        
        
        
    # Trains main network every step during episode
    def train(self, terminal_state):
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self._batch_size)
        

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        #current_states = current_states.reshape((self._batch_size,self._num_states))
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        #new_current_states = new_current_states.reshape((self._batch_size,self._num_states))
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), epochs=50, batch_size=16, verbose=0)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
    




