# OpenGym MagicSqaure-v0
# -------------------
#
# author: Pawel Gniewek, 2018

import random
import numpy as np
import math
import sys
import os

from keras import backend as K
import tensorflow as tf

import gym
import gym_magic

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.layers import *
from keras.optimizers import *

class Brain:
    """
    """

    def __init__(self, state_cnt, action_cnt):
        self.state_cnt = state_cnt
        self.action_cnt = action_cnt
        self.lr = 0.1

        self.model  = self._dnn()
        self.model_ = self._dnn()

        print( self.model.summary() )

    def _dnn(self):
        opt_ = Adam(lr=self.lr)
        model = Sequential() 
        model.add(Dense(64, input_dim=self.state_cnt, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_cnt, activation='softmax'))
        model.compile(loss='mse',optimizer=opt_, metrics=['mae'])       
        return model


    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)


    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)


    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_cnt), target=target).flatten()


    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()


        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])
            
            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)


    def run(self, agent):
        s = self.env.reset()
        R = 0 
        while True:            
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)
            
            if done: # terminal state
                s_ = None
            
            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r
 
            if done:
                print info
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
# Ensure we always get the same amount of randomness
# For tests and training only

np.random.seed(404)

PROBLEM = 'MagicSquare3x3-v0'
#PROBLEM = 'MagicSquare5x5-v0'
#PROBLEM = 'MagicSquare10x10-v0'
env = Environment(PROBLEM)
env.env.seed(404)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

dir_out = './model/'

if not os.path.exists( dir_out ):
    os.makedirs( dir_out )

try:
    print("RANDOM AGENT - FILLIN IN THE MEMORY")
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    print("AGENT - MODEL TRAINING")
    while True:
        env.run(agent)
        agent.brain.model.save(dir_out+PROBLEM + "-dqn.h5")
finally:
    agent.brain.model.save(dir_out+PROBLEM + "-dqn.h5")
