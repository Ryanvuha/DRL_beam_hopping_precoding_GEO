# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:41:58 2021

@author: mnguy
"""

#import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pylab as plt
import random
#import math
from collections import deque
import os
os.system('cls')  # For Windows
import numpy as np
#from os.path import dirname, join as pjoin
import scipy.io as sio
from parameters import parameters, write_para
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
para = write_para()



from trial_RL import DQNAgent

cwd = os.getcwd()
cwdchdir = os.path.normpath(os.getcwd() + os.sep + os.pardir)

###

mat_fname = 'C:/Project/Jamming/DataSave/H_orig.mat'
data = sio.loadmat(mat_fname)
H_orig = data['H_orig']

mat_fname = 'C:/Project/Jamming/DataSave/Hlong_est.mat'
data = sio.loadmat(mat_fname)
Hlong_est = data['Hlong_est']

mat_fname = 'C:/Project/Jamming/DataSave/HwJ_orig.mat'
data = sio.loadmat(mat_fname)
HwJ_orig = data['HwJ_orig']

mat_fname = 'C:/Project/Jamming/DataSave/HlongwJ_est.mat'
data = sio.loadmat(mat_fname)
HlongwJ_est = data['HlongwJ_est']


mat_fname = 'C:/Project/Jamming/DataSave/beam_codebook.mat'
data = sio.loadmat(mat_fname)
beam_codebook = data['beam_codebook'] # num_beam x num_antenna


mat_fname = 'C:/Project/Jamming/DataSave/beam_codebook_reduced.mat'
data = sio.loadmat(mat_fname)
beam_codebook_reduced = data['beam_codebook_reduced'] # num_beam x num_antenna

# convert channel in time point 0 to array
conv_channel = np.vstack(Hlong_est[0])

para.num_beam = beam_codebook.shape[0]
para.num_antenna = beam_codebook.shape[1]
para.num_user = conv_channel.shape[0]-1



# create signal strength

SigStr = [] # list has no shape attribute
for Tsym in range(H_orig.shape[0]):
    noise = np.random.randn(para.num_beam,para.num_user+1) + 1j*np.random.randn(para.num_beam,para.num_user+1)
    conv_channel = np.vstack(Hlong_est[Tsym])
    SigStr.append(np.matmul(beam_codebook,np.transpose(conv_channel))+noise) # num_beam x (num_user + 1): include Jammer
    

def create_recent_SigStr(SigStr, Tsym, RecentInd): # unbroadcast signal strength = 0
    SigStr1 = SigStr[Tsym]
    # RecentInd: vector indicate which beams are broadcast = action_recent return by DRL
    Recent_SigStr = np.zeros((para.num_beam, para.num_user, 2))
    for nb in RecentInd:
        nb = int(nb)
        for nu in range(para.num_user):
            Recent_SigStr[nb,nu, 0] = np.real(SigStr1[nb,nu])
            Recent_SigStr[nb,nu, 1] = np.imag(SigStr1[nb,nu])
    return np.reshape(Recent_SigStr,(para.num_beam *para.num_user*2,1))

def create_recent_SigStr_n(SigStr, Tsym, RecentInd, beam_codebook_reduced): # unbroadcast signal strength is neglected
    SigStr1 = SigStr[Tsym]
    bcr = beam_codebook_reduced
    # RecentInd: vector indicate which beams are broadcast = action_recent return by DRL
    Recent_SigStr = np.array([])
    Recent_beam = np.array([])
    for nb in RecentInd:
        nb = int(nb)
        for nu in range(para.num_user):
            SS_RI = np.append(np.real(SigStr1[nb,nu]),np.imag(SigStr1[nb,nu]))
            Recent_SigStr = np.append(Recent_SigStr, SS_RI)
        Recent_beam = np.append(Recent_beam,np.append(np.real(bcr[nb,:]), np.imag(bcr[nb,:])))

    Recent_SSandBeam = np.append(Recent_beam, Recent_SigStr)
    return Recent_SSandBeam



def create_state(Recent_SigStr, errorRate):
    State = np.append(Recent_SigStr, errorRate)
    return State
    

def init_NN_channel(x, y):
    # x is input, y is label
    model = Sequential()
    model.add(Dense(100, input_dim= x.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.fit(x, y, epochs=350, batch_size=2000, validation_split=0.2, verbose=1)
    model.save('model.h5')
    return model
    # loss='mean_absolute_error', loss='mean_squared_error', loss='hinge', loss='kullback_leibler_divergence'
    # loss='squared_hinge', loss='categorical_crossentropy', loss='sparse_categorical_crossentropy'


def NN_channel(x, y, model):
    model.fit(x, y, epochs=350, batch_size=2000, validation_split=0.1, verbose=1)
    model.save('model.h5')



N_initSampleNN = 2000
xNN = np.array([])
yNN = np.array([])
for Tsym in range(N_initSampleNN):
    RecentInd = np.array([])
    for nr in range(para.num_recent):
        RecentInd = np.append(RecentInd, 2**para.num_per_recent*nr + np.random.randint(2**para.num_per_recent))
    Recent_SSandBeam = create_recent_SigStr_n(SigStr, Tsym, RecentInd, beam_codebook_reduced)
    xNN = np.append(xNN, Recent_SSandBeam)
    conv_channel1 = np.vstack(Hlong_est[Tsym])
    conv_channel = np.reshape(conv_channel1,(conv_channel1.shape[0]*conv_channel1.shape[1],1))
    RIconv_channel = np.append(np.real(conv_channel), np.imag(conv_channel))       
    yNN = np.append(yNN, RIconv_channel)

xNN = np.reshape(xNN,(N_initSampleNN, int(xNN.shape[0]/N_initSampleNN)))
yNN = np.reshape(yNN,(N_initSampleNN, int(yNN.shape[0]/N_initSampleNN)))

val_norm_x = np.linalg.norm(xNN,'fro')/N_initSampleNN*35
#val_norm_y = np.linalg.norm(yNN,'fro')/N_initSampleNN

xNN = xNN/val_norm_x
#yNN = yNN/val_norm_y
model = init_NN_channel(xNN, yNN)

ypre = model.predict(xNN)
ypre[0]   











num_actions = para.num_actions
num_states = para.num_states
num_episodes = 10000
STATS_EVERY = 200

# initialize


reward_store = np.array([])
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

start_cnt = 1
RecentInd = np.array([])
for nr in range(para.num_recent):
    RecentInd = np.append(RecentInd, 2**para.num_per_recent*nr + np.random.randint(2**para.num_per_recent))
errorRate = 10

agent = DQNAgent() 
for cnt1 in range(start_cnt,num_episodes):
    
    Recent_SigStr = create_recent_SigStr(SigStr, Tsym, RecentInd)    
    current_state = create_state(Recent_SigStr, errorRate)
    
    
    allpredict = agent.model.predict(np.array([current_state]))
    allpredict_index = allpredict[0].argsort()[::-1] # sort in decreasing order
    action_int = allpredict_index[0]
    action_indicate, action_recent, action_refine = agent.convert_array_action(action_int)
    
    RecentInd = action_recent
    
    # compute reward and error rate here
    errorRate = 5*abs(np.random.randn())
    reward = 5*abs(np.random.randn())
    ##-
    
    Recent_SigStr = create_recent_SigStr(SigStr, Tsym, RecentInd)    
    new_state = create_state(Recent_SigStr, errorRate) 
    
     
     
    
                
    
    # get a reward
    

    reward_store = np.append(reward_store, reward)
     
    done = True
    agent.update_replay_memory((current_state, action_int, reward, new_state, done))

    if cnt1 > 200:
        agent.train(True) 
        
        
    if cnt1 % 1 == 0:
        #print('Episode {} of {}'.format(cnt1+1, num_episodes))
        #print('Action of Episode {} is {} and {}'.format(cnt1+1, action))
        print('Reward of Episode {} is {}'.format(cnt1+1, reward))
        #print('Random-action:',action_random)
        #print('Prediction of safe-condition before adding new sample',predictions_before)
        #print('Prediction of safe-condition after adding new sample',agent.safe_predict(state_action))
        
    if cnt1 <= STATS_EVERY:                                      
        aggr_ep_rewards['ep'].append(cnt1)                
        aggr_ep_rewards['avg'].append(sum(reward_store)/STATS_EVERY)                
        aggr_ep_rewards['max'].append(max(reward_store))                
        aggr_ep_rewards['min'].append(min(reward_store)) 

    if cnt1 > STATS_EVERY:                                      
        aggr_ep_rewards['ep'].append(cnt1)                
        aggr_ep_rewards['avg'].append(sum(reward_store[-STATS_EVERY:])/STATS_EVERY)                
        aggr_ep_rewards['max'].append(max(reward_store[-STATS_EVERY:]))                
        aggr_ep_rewards['min'].append(min(reward_store[-STATS_EVERY:])) 
        
    if cnt1 > 1  and not cnt1%5: 
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
        plt.show()  
    
    if not cnt1%20:
        replay2 = agent.replay_memory
        file_Name = "replay2"
        fileObject = open(file_Name,'wb') 
        pickle.dump(replay2,fileObject)  
        # here we close the fileObject
        fileObject.close()
        
        file_Name = "aggr_ep_rewards"
        fileObject = open(file_Name,'wb') 
        pickle.dump(aggr_ep_rewards,fileObject)  
        # here we close the fileObject
        fileObject.close()
        
        file_Name = "reward_store"
        fileObject = open(file_Name,'wb') 
        pickle.dump(reward_store,fileObject)  
        # here we close the fileObject
        fileObject.close()
        
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=1)
plt.show()
    
    # slicing pronlem here
    
    
    
    